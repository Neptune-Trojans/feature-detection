import cv2
import matplotlib.pyplot as plt
import numpy as np


class SIFTPlayGround:
    def __init__(self):
        self.image = cv2.imread('../images/image01.jpg')
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        self.reference_img = self._generate_test_image()

    def _generate_test_image(self):
        image = cv2.rotate(self.image, cv2.ROTATE_180)
        cv2.imwrite('../images/ref_image01.jpg', image)
        return image

    def plot_orig_vs_reference(self):
        fx, plots = plt.subplots(1, 2, figsize=(20, 10))

        plots[0].set_title("Original Image")
        plots[0].imshow(self.image)

        plots[1].set_title("Reference Image")
        plots[1].imshow(self.reference_img)
        plt.savefig('../images/orig_vs_reference.jpg')

    def plot_key_points(self, img, keypoints, path):

        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        keypoints_without_size = np.copy(img)
        keypoints_with_size = np.copy(img)

        cv2.drawKeypoints(img, keypoints, keypoints_without_size, color=(0, 255, 0))
        cv2.drawKeypoints(img, keypoints, keypoints_with_size, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        fx, plots = plt.subplots(1, 2, figsize=(20, 10))

        plots[0].set_title("Train keypoints With Size")
        plots[0].imshow(keypoints_with_size, cmap='gray')

        plots[1].set_title("Train keypoints Without Size")
        plots[1].imshow(keypoints_without_size, cmap='gray')
        plt.savefig(path)

    def find_key_points(self):
        sift = cv2.xfeatures2d.SIFT_create()
        org_gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        ref_gray = cv2.cvtColor(self.reference_img, cv2.COLOR_RGB2GRAY)

        org_keypoints, org_descriptor = sift.detectAndCompute(org_gray, None)
        ref_keypoints, ref_descriptor = sift.detectAndCompute(ref_gray, None)

        print("Number of Key Points Detected In Original IMG {}".format(len(org_keypoints)))
        print("Number of Key Points Detected In Reference IMG {}".format(len(ref_keypoints)))

        return (org_keypoints, org_descriptor), (ref_keypoints, ref_descriptor)

    def key_points_matcher(self, orgin_desc, dest_desc):

        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        matches = bf.match(orgin_desc, dest_desc)

        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def draw_matched_points(self, org_key_points, ref_key_points, matches):
        ref_img = np.copy(self.reference_img)

        result = cv2.drawMatches(self.image, org_key_points, self.reference_img, ref_key_points, matches, ref_img, flags=2)

        # Display the best matching points
        plt.figure(figsize=(14.0, 7.0))
        # plt.rcParams['figure.figsize'] = [14.0, 7.0]
        plt.title('Best Matching Points')
        plt.imshow(result)
        plt.savefig('../images/matched.jpg')


if __name__ == '__main__':
    # https://aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/
    # https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
    sift = SIFTPlayGround()
    sift.plot_orig_vs_reference()
    (org_keypoints, org_descriptor), (ref_keypoints, ref_descriptor) = sift.find_key_points()
    sift.plot_key_points(sift.image, org_keypoints, '../images/orig_key_points.jpg')
    sift.plot_key_points(sift.reference_img, ref_keypoints, '../images/ref_key_points.jpg')
    matches = sift.key_points_matcher(org_descriptor, ref_descriptor)
    print('total found matches {}'.format(len(matches)))
    sift.draw_matched_points(org_keypoints, ref_keypoints, matches)
