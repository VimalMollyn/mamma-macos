import trimesh
import numpy as np
np.random.seed(0)
import os
import glob
import torch


torch.set_default_dtype(torch.float32)


def _load_sdf_backend():
    """Import ``pytorch_sdf`` lazily, with a clear error if it's missing.

    ``pytorch_sdf`` ships CUDA kernels and is only exercised by the SDF-based
    contact/penetration loss (active when ``--part-mesh`` points at .ply
    files). Importing it at module load would break CPU/MPS-only installs
    (e.g. macOS, no CUDA toolkit) that never use that loss, so we defer it to
    first use.
    """
    try:
        from pytorch_sdf import sdf
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise ImportError(
            "pytorch_sdf is required for the SDF contact/penetration loss but "
            "is not installed (it builds CUDA kernels and is unavailable on "
            "non-CUDA platforms). Run without the SDF loss — leave --part-mesh "
            "unset — or install pytorch_sdf on a CUDA host."
        ) from exc
    return sdf


def get_3d_bbox(vertices,):

    # Compute the 3d bbox
    min_coords = torch.min(vertices, dim=1)[0]
    max_coords = torch.max(vertices, dim=1)[0]
    bbox = torch.stack([min_coords, max_coords], dim=1)
    return bbox

def get_iou(bbox1: torch.tensor, bbox2: torch.tensor):
    intersection = torch.max(torch.min(bbox1[:, 1], bbox2[:, 1]) - torch.max(bbox1[:, 0], bbox2[:, 0]),
                             torch.zeros_like(bbox1[:, 0]))
    union = torch.max(bbox1[:, 1], bbox2[:, 1]) - torch.min(bbox1[:, 0], bbox2[:, 0])
    iou = torch.prod(intersection, dim=1) / torch.prod(union, dim=1)
    return iou


class MultiSDF:
    def __init__(self, mesh_faces, part_mesh_path=None, max_batch_size=4, valid_vertices_idx=None, distance_method="bvh"):
        # part_mesh_path is optional. When None or empty the SDF-based
        # auxiliary loss is effectively disabled (no .ply files to load).
        # The caller (OptimizeSMPLX) supplies it from PathsConfig built
        # from argparse (--part-mesh) at the run_ma_3d.py entry point.
        self.distance_method = distance_method
        self.part_mesh_path = (
            glob.glob(f"{part_mesh_path}/*.ply") if part_mesh_path else []
        )
        self.mesh_faces = mesh_faces
        self.fixed_mesh = None
        self.part_meshes = None
        self.total_sdf = None
        self.source_mesh = None
        # Built lazily on first SDF evaluation so MultiSDF can be constructed
        # (and ma_3d can run without the SDF loss) on hosts lacking pytorch_sdf.
        self._sdf_fn = None
        self._sdf_unavailable = False  # set once if pytorch_sdf can't be imported
        self.face_masks = {}
        self.sdf_part_params = {}
        self.max_batch_size = max_batch_size
        self.valid_vertices_idx = valid_vertices_idx.astype(bool)
        self.mesh_faces = np.load("smplx_simplified_face_idx.npy").astype(np.int32)


    def _ensure_sdf(self):
        """Return the pytorch_sdf evaluator, or ``None`` if it's unavailable.

        Deferred so importing/instantiating MultiSDF never requires the
        CUDA-only ``pytorch_sdf`` package. When it can't be imported (e.g. on
        macOS / any non-CUDA host) we disable the SDF contact/penetration term
        gracefully — returning ``None`` so callers skip it — instead of
        crashing the whole ma_3d optimization over an optional refinement loss.
        """
        if self._sdf_fn is None and not self._sdf_unavailable:
            try:
                self._sdf_fn = _load_sdf_backend().SDF(
                    distance_method=self.distance_method
                ).sdf_with_winding_numbers
            except ImportError as exc:
                self._sdf_unavailable = True
                print(f"[MultiSDF] SDF contact/penetration loss disabled: {exc}")
        return self._sdf_fn

    @property
    def sdf(self):
        """The pytorch_sdf winding-number evaluator, built on first use."""
        return self._ensure_sdf()


    def multi_people_sdf_loss(self, bodies_vertices, bodies_sampled_verts=None, bodies_contacts=None, ignore_fist_t_frames=10, weight=1.):
        if self._ensure_sdf() is None:
            # pytorch_sdf unavailable (non-CUDA host) — skip this term.
            return weight * torch.tensor(0.0, device=bodies_vertices[0].device)
        total_loss = torch.tensor(0.0).to(bodies_vertices[0].device)
        min_batch_size = self.max_batch_size

        for main_body_id, body_vertices in enumerate(bodies_vertices):

            bbox_fixed_body = get_3d_bbox(body_vertices)
            body_triangles = body_vertices[:, self.mesh_faces].detach()


            for body_id, body_vertices_other in enumerate(bodies_vertices):
                if body_id == main_body_id:
                    continue
                bbox_other_body = get_3d_bbox(body_vertices_other)
                iou = get_iou(bbox_fixed_body, bbox_other_body)

                valid_iou = iou > 0.1
                valid_iou[:ignore_fist_t_frames] = False
                if not valid_iou.any().item():
                    continue

                if bodies_contacts is not None:
                    contact_weights = bodies_contacts[body_id][valid_iou, :].contiguous()
                    a, b = 0.2, 0.6
                    contact_weights = torch.clamp((contact_weights - a) / (b - a), min=0., max=1.)
                else:
                    contact_weights = 1.


                body_triangles_new = body_triangles[valid_iou].contiguous()
                if bodies_sampled_verts is not None:
                    body_vertices_other_new = bodies_sampled_verts[body_id][valid_iou].contiguous()

                else:
                    body_vertices_other_new = body_vertices_other[valid_iou][:, self.valid_vertices_idx].contiguous()

                total_batch_size = body_triangles_new.shape[0]
                start_idx, end_idx = [], []
                batch_size = min(min_batch_size, total_batch_size)
                for i in range(0, total_batch_size, batch_size):
                    start_idx.append(i)
                    end_idx.append(i+batch_size)
                end_idx[-1] = total_batch_size
                start_end_idx = list(zip(start_idx, end_idx))

                for start_b, end_b in start_end_idx:

                    min_dist, inside = self.sdf(body_vertices_other_new[start_b: end_b].contiguous(),
                                                body_triangles_new[start_b: end_b].contiguous(), None)
                    w_contact = contact_weights[start_b: end_b].contiguous() if isinstance(contact_weights, torch.Tensor) else contact_weights
                    penetration_margin = -0.002 #0.01 #0.05
                    close_mask = (min_dist.abs() < 0.01) & (min_dist > 0.0) # only force contact
                    w_contact = w_contact * close_mask.float()
                    if w_contact.sum().item() > 0:
                        contact_radius = 0.001 #0.015  # ~1.5 cm
                        d = min_dist.clamp(min=0.0)
                        contact_residual = torch.relu(d - contact_radius)  # only penalize if d > r
                        loss_contact = (w_contact * contact_residual.pow(2)).sum() / (w_contact.sum() + 1e-8)
                    else:
                        loss_contact = torch.tensor(0.0).to(body_vertices.device)
                    if inside.sum().item() == 0:
                            loss_repulsion = torch.tensor(0.0).to(body_vertices.device)
                    else:
                        loss_repulsion = torch.nn.functional.relu(penetration_margin - min_dist[inside]).pow(2).mean()
                    total_loss = total_loss + 0.2*loss_contact + loss_repulsion
        return weight*total_loss


    def _segment_mesh_into_parts(self, source_mesh):
        self.part_meshes = {}
        for part_mesh in self.part_mesh_path:
            mesh_name = os.path.basename(part_mesh).split(".")[0]
            full_mesh = source_mesh.copy()
            if mesh_name not in self.face_masks:
                mesh = trimesh.load_mesh(part_mesh)
                face_mask = mesh.visual.face_colors[:, 0] == 255
                self.face_masks[mesh_name] = np.array(face_mask, dtype=bool)
            else:
                face_mask = self.face_masks[mesh_name]
            full_mesh.update_faces(face_mask)
            full_mesh.remove_unreferenced_vertices()
            self.part_meshes[mesh_name] = full_mesh


    def _segment_mesh_batch(self, source_verts_np):
        vertices = {}
        faces = {}
        for enum, source_verts in enumerate(source_verts_np):
            source_mesh = trimesh.Trimesh(source_verts, self.mesh_faces)
            for part_mesh in self.part_mesh_path:
                mesh_name = os.path.basename(part_mesh).split(".")[0]
                full_mesh = source_mesh.copy()
                face_mask = self.face_masks[mesh_name]
                full_mesh.update_faces(face_mask)
                full_mesh.remove_unreferenced_vertices()
                if mesh_name not in vertices:
                    vertices[mesh_name] = []
                    faces[mesh_name] = []
                vertices[mesh_name].append(full_mesh.vertices)
                faces[mesh_name].append(full_mesh.vertices[full_mesh.faces])


        for mesh_name in vertices.keys():
            vertices[mesh_name] = np.asarray(vertices[mesh_name])
            faces[mesh_name] = np.asarray(faces[mesh_name])
        return vertices, faces


    def _create_fixed_mesh(self, source_verts_np):
        self.total_sdf = np.zeros(source_verts_np.shape[0])
        if self.source_mesh is None:
            self.source_mesh = trimesh.Trimesh(source_verts_np, self.mesh_faces)
        else:
            self.source_mesh.vertices = source_verts_np


    def batch_sdf_loss_pt(self, source_verts_batch, start_b, end_b, triangles_full, verts_full):
        points = source_verts_batch[start_b:end_b].contiguous()
        with torch.no_grad():
            triangles = triangles_full[start_b:end_b]
        min_dist, inside = self.sdf(points, triangles, None)
        return (-min_dist[inside]).sum()


    def batch_multi_sdf_loss(self, source_verts_batch, weight=1.):
        if self._ensure_sdf() is None:
            # pytorch_sdf unavailable (non-CUDA host) — skip this term.
            return weight * torch.tensor(0.0, device=source_verts_batch.device)
        total_loss = torch.tensor(0.0, device=source_verts_batch.device, requires_grad=True)
        source_verts_np = source_verts_batch.detach().cpu().numpy()
        self._create_fixed_mesh(source_verts_np[-1])
        self._segment_mesh_into_parts(self.source_mesh)
        part_verts, part_triangles, = self._segment_mesh_batch(source_verts_np)
        for part_name in part_verts:
            verts_np = part_verts[part_name]
            triangles_np = part_triangles[part_name]

            vertices = torch.from_numpy(verts_np).float().to('cuda')
            triangles = torch.from_numpy(triangles_np).float().to('cuda')

            total_batch_size = source_verts_batch.shape[0]

            start_idx, end_idx = [], []
            batch_size = min(self.max_batch_size,total_batch_size)
            for i in range(0, total_batch_size, batch_size):
                start_idx.append(i)
                end_idx.append(i+batch_size)
            end_idx[-1] = total_batch_size
            start_end_idx = list(zip(start_idx, end_idx))
            for start_b, end_b in start_end_idx:
                sub_loss = self.batch_sdf_loss_pt(source_verts_batch, start_b, end_b, triangles, vertices,)
                total_loss = total_loss + sub_loss
        return weight*total_loss
