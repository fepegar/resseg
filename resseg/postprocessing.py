import torchio as tio
import SimpleITK as sitk


# # TODO: make TorchIO simpler so this is possible
# class LargestConnectedComponent(tio.transforms.preprocessing.label.label_transform.LabelTransform):
#     def apply_transform(self, subject):
#         for image in self.get_images(subject):
#             sitk_image = image.as_sitk()
#             largest_cc = self.keep_largest_cc(sitk_image)
#             tensor, _ = self.sitk_to_nib(largest_cc)
#             image.set_data(tensor)
#         return subject

#     @staticmethod
#     def keep_largest_cc(image):
#         connected_components = sitk.ConnectedComponent(image)
#         labeled_cc = sitk.RelabelComponent(connected_components)
#         largest_cc = labeled_cc == 1
#         return largest_cc



def keep_largest_cc(image):
    sitk_image = image.as_sitk()
    connected_components = sitk.ConnectedComponent(sitk_image)
    labeled_cc = sitk.RelabelComponent(connected_components)
    largest_cc = labeled_cc == 1
    tensor, _ = tio.io.sitk_to_nib(largest_cc)
    image.set_data(tensor)
