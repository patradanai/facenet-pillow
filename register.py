import cv2
import align.detect_face
import tensorflow as tf

npy = './align'
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 44
frame_interval = 3
batch_size = 1000
image_size = 183
input_image_size = 160

# init Cv2

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        cap = cv2.VideoCapture(0)
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, npy)
        while True:
            ret, frame = cap.read()

            bounding_boxes, points = align.detect_face.detect_face(
                frame, minsize, pnet, rnet, onet, threshold, factor)

            if ret:
                for i in range(len(bounding_boxes)):
                    # print(bounding_boxes[i][0], bounding_boxes[i][1])
                    cv2.rectangle(frame, (int(bounding_boxes[i][0]), int(bounding_boxes[i][1])), (
                        int(bounding_boxes[i][2]), int(bounding_boxes[i][3])), (0, 255, 0), 2)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
