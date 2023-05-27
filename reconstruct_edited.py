#!/usr/bin/env python3
import numpy as np

#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch
import deep_sdf.utils
import deep_sdf
import deep_sdf.workspace as ws

def pixel_to_world(x, y, z, camera_pos):
    # Convert pixel coordinates to world coordinates
    world_x = (x - camera_pos[0]) * z / camera_pos[2]
    world_y = (y - camera_pos[1]) * z / camera_pos[2]
    world_z = z

    return (world_x, world_y, world_z)

def raycast(decoder, latent_vec, filename):
    decoder.eval()
    camera_position = np.array([0.0, 0.0, 1.0])  # Adjust the camera position accordingly
    image_width = 640
    image_height = 480
    focal_length = 500  # Adjust the focal length of the camera accordingly
    z_increment = -0.01  # Increment of z value for each pixel
    table_z = -1.0
    depth_image = np.zeros((image_height, image_width), dtype=np.float32)
    
    latent_repeat = latent_vec.repeat(image_height * image_width, 1)  # Repeat latent vector
    xyz = np.zeros((image_height * image_width, 3), dtype=np.float32)
    
    # Generate the xyz coordinates for all pixels
    for y in range(image_height):
        for x in range(image_width):
            point_3d = np.array([x - image_width/2, y - image_height/2, focal_length])
            point_world = camera_position + (point_3d / np.linalg.norm(point_3d))
            xyz[y * image_width + x] = point_world
    
    latent_repeat = latent_repeat.cuda()
    xyz = torch.from_numpy(xyz).cuda()
    
    # Concatenate latent_repeat and xyz along the second dimension
    inputs = torch.cat([latent_repeat, xyz], dim=1)
    
    # Calculate the SDF values using the DeepSDF decoder
    sdf_values = decoder(inputs)
    
    # Iterate over each pixel in the image
    for y in range(image_height):
        for x in range(image_width):
            z_value = camera_position[2]  # Starting z value is the camera position
            prev_sdf_val = 1000
            while True:
                sdf_value = sdf_values[y * image_width + x]
                
                # Check if the SDF value indicates the object surface is crossed
                if sdf_value < 0.0:
                    break
                
                if sdf_value > prev_sdf_val:
                    z_value = table_z
                    break
                
                z_value += z_increment
                prev_sdf_val = sdf_value
            
            # Store the z value (depth) in the depth image
            depth_image[y, x] = z_value

    # Save the depth image as an npy file
    np.save(filename+'.npy', depth_image)

def find_bounding_box(decoder, latent_vec, camera_position, voxel_size, image_size, margin):
    device = latent_vec.device

    # Set the z value for sampling the xy plane
    z_value = 0.5

    # Create a grid of x and y coordinates
    x_coords = torch.linspace(
        camera_position[0] - (image_size[0] // 2) * voxel_size,
        camera_position[0] + (image_size[0] // 2) * voxel_size,
        image_size[0]
    ).to(device)
    y_coords = torch.linspace(
        camera_position[1] - (image_size[1] // 2) * voxel_size,
        camera_position[1] + (image_size[1] // 2) * voxel_size,
        image_size[1]
    ).to(device)

    # Generate the mesh grid
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords)
    xy_coords = torch.stack([x_grid, y_grid], dim=2).reshape(-1, 2)

    # Set the z value for the xy coordinates
    z_coords = torch.full((image_size[0] * image_size[1],), z_value, device=device)

    # Concatenate the coordinates
    sample_coords = torch.stack([xy_coords[:, 0], xy_coords[:, 1], z_coords], dim=1)

    # Calculate the SDF values for the sample coordinates
    sdf_values = deep_sdf.utils.decode_sdf(decoder, latent_vec, sample_coords)

    # Find the indices with SDF values close to zero
    close_indices = torch.where(torch.abs(sdf_values) < margin)[0]

    if len(close_indices) > 0:
        # Get the minimum and maximum x and y indices
        min_x, max_x = torch.min(xy_coords[close_indices, 0]), torch.max(xy_coords[close_indices, 0])
        min_y, min_x = torch.min(xy_coords[close_indices, 1]), torch.max(xy_coords[close_indices, 1])
    else:
        # If no indices are close to zero, set the bounding box to the whole image size
        min_x, max_x = 0, image_size[0] - 1
        min_y, max_y = 0, image_size[1] - 1

    return min_x, max_x, min_y, max_y

def raycast2(decoder, latent_vec, filename):
    decoder.eval()

    device = latent_vec.device

    camera_position = torch.tensor([0.0, 0.0, -2.0])  # Update with your camera position
    image_size = (640, 480)  # Update with your image size
    voxel_size = 0.01  # Update with your desired voxel size
    margin = 1 
    # Calculate the voxel origin based on the image size and voxel size
    voxel_origin = [
        camera_position[0] - voxel_size * (image_size[0] // 2),
        camera_position[1] - voxel_size * (image_size[1] // 2),
        camera_position[2]
    ]

    sdf_values = torch.zeros(image_size).to(device) 
    depth_image = np.zeros(image_size, dtype=np.float32)

    # Find the bounding box for the xy plane
    min_x, max_x, min_y, max_y = find_bounding_box(decoder, latent_vec, camera_position, voxel_size, image_size, margin)
    print(min_x, max_x, min_y, max_y)
    for y in range(min_y, max_y + 1):
      for x in range(min_x, max_x + 1):
        # print(x,y)
        # Calculate the ray direction for the current pixel
        ray_direction = torch.tensor([
            (x * voxel_size) + voxel_origin[0],
            (y * voxel_size) + voxel_origin[1],
            voxel_origin[2]
        ]) - camera_position

        ray_direction = ray_direction / torch.norm(ray_direction)

        # Perform ray marching until SDF value becomes negative or starts to rise again
        current_position = camera_position.clone().to(device)
        prev_sdf_value = float('inf')
        step_size = voxel_size / 2.0

        while True:
            # Calculate the SDF value at the current position
            sdf_value = deep_sdf.utils.decode_sdf(decoder, latent_vec, current_position.unsqueeze(0)).item()

            # Store the SDF value for the current pixel
            sdf_values[x, y] = sdf_value

            # Store the z value (depth) in the depth image
            depth_image[x, y] = current_position[2].item()

            # Check if the SDF value is less than zero or starts to rise again
            if sdf_value < 0 or sdf_value > prev_sdf_value:
                
                break

            prev_sdf_value = sdf_value
            current_position += ray_direction.to(device) * step_size

                

    # Save the depth image as an npy file
    np.save(filename + '.npy', depth_image)

def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs, xyz], 1).cuda()

        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent

def raycast3(decoder, latent_vec, filename):
    decoder.eval()
    # Example usage

    image_width = 640
    image_height = 480
    voxel_resolution = 256
    camera_distance = 5.0
    # Set up parameters
    bounding_box = [-1, 1]
    voxel_size = (bounding_box[1] - bounding_box[0]) / voxel_resolution
    ray_direction = torch.tensor([0, 0, -1])  # Assuming the camera is looking straight down the negative z-axis
    camera_position = torch.tensor([0, 0, camera_distance])

    # Initialize depth image
    depth_image = np.zeros((image_height, image_width))

    # Define the range of pixels to search within the middle region of the image
    middle_region_width = image_width // 4  # Adjust this value to change the width of the middle region
    middle_region_start_x = (image_width - middle_region_width) // 2
    middle_region_end_x = middle_region_start_x + middle_region_width

    middle_region_height = image_height // 4  # Adjust this value to change the height of the middle region
    middle_region_start_y = (image_height - middle_region_height) // 2
    middle_region_end_y = middle_region_start_y + middle_region_height

    # Calculate SDF values for each pixel
    for y in range(middle_region_start_y, middle_region_end_y):
        for x in range(middle_region_start_x, middle_region_end_x):
            pixel_position = torch.tensor([
                (x - image_width / 2) * voxel_size,
                (y - image_height / 2) * voxel_size,
                camera_distance
            ])

            sdf_value = None  # Initial SDF value at camera position
            prev_sdf_value = float('inf')
            # Perform raycasting-like method
            while True:
                world_position = camera_position + pixel_position
                sample = torch.tensor(world_position, dtype=torch.float32).unsqueeze(0).cuda()
                sdf_value = deep_sdf.utils.decode_sdf(decoder, latent_vec, sample)
                pixel_position += voxel_size * ray_direction
                if sdf_value < 0 or sdf_value > prev_sdf_value:            
                    break
                prev_sdf_value = sdf_value

            # Store the sdf_value at the last valid position (when it becomes negative)
            depth_image[x, y] = sdf_value.item()

    # Save the depth image as an npy file
    np.save(filename + '.npy', depth_image)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)

    random.shuffle(npz_filenames)

    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    for ii, npz in enumerate(npz_filenames):

        if "npz" not in npz:
            continue

        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)

        logging.debug("loading {}".format(npz))

        data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, npz[:-4] + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + "-" + str(k + rerun) + ".pth"
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + ".pth"
                )

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
                and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(npz))

            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            start = time.time()
            try:
                err, latent = reconstruct(
                    decoder,
                    int(args.iterations),
                    latent_size,
                    data_sdf,
                    0.01,  # [emp_mean,emp_var],
                    0.1,
                    num_samples=800,
                    lr=5e-3,
                    l2reg=True,
                )
            except:
                print('to few samples')
                continue
            logging.debug("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

            decoder.eval()

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    # deep_sdf.mesh.create_mesh(
                    #     decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18)
                    # )

                    raycast2(decoder, latent, mesh_filename)
                    
                print("total time: {}".format(time.time() - start))

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)



    