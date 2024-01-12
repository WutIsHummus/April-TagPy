import copy
import time
import argparse

import cv2 as cv
import numpy as np  
from pupil_apriltags import Detector

fx, fy = 1430, 1430
cx, cy = 480, 620
from ui_script import start_ui_thread, driveMap, get_drive_map

start_ui_thread()
 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)

    parser.add_argument("--families", type=str, default='tag36h11')
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--quad_decimate", type=float, default=2.0)
    parser.add_argument("--quad_sigma", type=float, default=0.0)
    parser.add_argument("--refine_edges", type=int, default=0.25)
    parser.add_argument("--decode_sharpening", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0)

    

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    families = args.families
    nthreads = args.nthreads
    quad_decimate = args.quad_decimate
    quad_sigma = args.quad_sigma
    refine_edges = args.refine_edges
    decode_sharpening = args.decode_sharpening
    debug = args.debug
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )
    elapsed_time = 0
    while True:
        current_drive_map = get_drive_map()
        print(current_drive_map)
        start_time = time.time()
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            image,
            estimate_tag_pose=True,
            camera_params=[fx, fy, cx, cy],
            tag_size=0.06,
        )
        debug_image = draw_tags(debug_image, tags, elapsed_time)
        if current_drive_map:
                tag_positions = []  
                tag_centers = []  
                for tag_id in current_drive_map:
                    for tag in tags:
                        if tag.tag_id == tag_id:
                            pose_t = tag.pose_t  
                            center = (int(tag.center[0]), int(tag.center[1])) 
                            tag_positions.append(pose_t)
                            tag_centers.append(center)
                            break 

                for i in range(len(tag_centers) - 1):
                    distance_3d = np.linalg.norm(tag_positions[i] - tag_positions[i+1])
                    mid_point = ((tag_centers[i][0] + tag_centers[i+1][0]) // 2, (tag_centers[i][1] + tag_centers[i+1][1]) // 2)
                    cv.putText(debug_image, f'{distance_3d:.2f}', mid_point, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv.line(debug_image, tag_centers[i], tag_centers[i+1], (0, 255, 0), 2)



        elapsed_time = time.time() - start_time
        cv.imshow('AprilTag Detect Demo', debug_image)
        key = cv.waitKey(1)
        if key == 27:  
            break
        

    cap.release()
    cv.destroyAllWindows()


def draw_tags(
    image,
    tags,
    elapsed_time,
):
    for tag in tags:
        tag_family = tag.tag_family
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners
        pose_t = tag.pose_t
        distance_to_tag = np.linalg.norm(pose_t)

        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))
        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)
        cv.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)
        cv.putText(image,
                   str(distance_to_tag) + ':' + str(tag_id),
                   (corner_01[0], corner_01[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)
    cv.putText(image,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()