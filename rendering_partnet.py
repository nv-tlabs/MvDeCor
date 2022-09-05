# SPDX-FileCopyrightText: Copyright (c) <2022> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import h5py
from src.render_utils import *
import os
import json
from scipy import stats
import time

category = sys.argv[1]
set_type = sys.argv[2]
start = int(sys.argv[3])
end = int(sys.argv[4])

# "partnet/partnet_dataset/sem_seg_h5/"
part_net_root_path = sys.argv[5] + "{}-3/".format(category)

# "partnet/partnet_dataset/data_v0/"
partnet_dir = sys.argv[6]


def load_partnet(input_objs_dir):
    def load_obj(fn):
        fin = open(fn, 'r')
        lines = [line.rstrip() for line in fin]
        fin.close()

        vertices = [];
        faces = [];
        for line in lines:
            if line.startswith('v '):
                vertices.append(np.float32(line.split()[1:4]))
            elif line.startswith('f '):
                faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

        return np.vstack(vertices), np.vstack(faces)

    vs = [];
    fs = [];
    vid = 0;
    all_obj_files = os.listdir(input_objs_dir)
    all_obj_files.sort()
    face_ids = []
    fid = 0
    for item in all_obj_files:
        if item.endswith('.obj'):
            cur_vs, cur_fs = load_obj(os.path.join(input_objs_dir, item))
            vs.append(cur_vs)
            fs.append(cur_fs + vid)
            face_ids.append(fid)
            vid += cur_vs.shape[0]
            fid += cur_fs.shape[0]
    v_arr = np.concatenate(vs, axis=0)
    v_arr_ori = np.array(v_arr, dtype=np.float32)
    f_arr = np.concatenate(fs, axis=0)
    tmp = np.array(v_arr[:, 0], dtype=np.float32)
    v_arr[:, 0] = v_arr[:, 2]
    v_arr[:, 2] = -tmp
    partnet_mesh = trimesh.Trimesh(vertices=v_arr_ori, faces=f_arr - 1)
    return partnet_mesh, face_ids


datas = []
labels = []
Points = []
Labels = []

try:
    for i in range(100):
        with h5py.File(part_net_root_path + "{}-0{}.h5".format(set_type, i),
                       "r") as hf:
            train_points = np.array(hf.get("data"))
            train_labels = np.array(hf.get("label_seg"))
        Points.append(train_points)
        Labels.append(train_labels)
except:
    pass

Points = np.concatenate(Points, 0)
Labels = np.concatenate(Labels, 0)

output_path = "partnet/renderings/{}/".format(category)
colors = np.random.random((100, 3))

with open("partnet/partnet_dataset/stats/train_val_test_split/{"
          "}.{}.json".format(category, set_type), "r") as file:
    data = json.load(file)

for model_index, model_name in enumerate(data):
    if (model_index >= start) and (model_index <= end):
        pass
    else:
        continue

    t1 = time.time()
    anno_id = model_name["anno_id"]
    model_name = model_name["model_id"]
    camera_poses = create_random_uniform_camera_poses(2.0, low_scale=0.7)
    render = Render(size=256, camera_poses=camera_poses)
    print(model_index, model_name, flush=True)

    mesh, face_ids = load_partnet(partnet_dir + anno_id + "/objs/")

    os.makedirs(output_path + model_name + "/", exist_ok=True)
    points = Points[model_index]
    labels = Labels[model_index]
    face_labels, _, mesh = transfer_labels_shapenet_points_to_mesh_partnet(points, labels.astype(np.int32), mesh)

    # Majority voting within each segment to remove the noise that might occur because
    # of transfer of labels from points to mesh.
    for ind, f in enumerate(face_ids[0:-1]):
        face_labels[face_ids[ind]: face_ids[ind + 1]] = stats.mode(face_labels[face_ids[ind]: face_ids[ind + 1]])[0]
    face_labels[face_ids[ind + 1]:] = stats.mode(face_labels[face_ids[ind + 1]:])[0]

    trimesh.exchange.export.export_mesh(mesh, "{}mesh.obj".format(output_path + model_name + "/"))
    np.save("{}faceid.npy".format(output_path + model_name + "/"), face_labels)
    print("Time taken: ", time.time() - t1)
    (
        triangle_ids,
        rendered_images,
        normal_maps,
        depth_images,
        p_images,
    ) = render.render(
        mesh,
        clean=False,
    )

    for i in range(len(rendered_images)):
        np.save(
            "{}tri_image_{}.npy".format(output_path + model_name + "/", i),
            triangle_ids[i],
        )
        np.save(
            "{}p_image_{}.npy".format(output_path + model_name + "/", i),
            p_images[i].astype(np.float32),
        )
        np.save(
            "{}rendered_image_{}.npy".format(output_path + model_name + "/", i),
            rendered_images[i],
        )
        np.save(
            "{}normal_image_{}.npy".format(output_path + model_name + "/", i),
            normal_maps[i].astype(np.float16),
        )
        np.save(
            "{}depth_image_{}.npy".format(output_path + model_name + "/", i),
            depth_images[i],
        )
