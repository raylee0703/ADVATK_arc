# import numpy as np
# from PIL import Image
# from utils.magnet.worker import slice_data_for_arc

# img = Image.open('dirty_chicken_Gray.jpg')
# img = np.array(img)
# img = np.expand_dims(img, axis=0)
# img = np.expand_dims(img, axis=0)
# #print(img.shape)
# img_set = slice_data_for_arc(img, 40, 40)
# #print(img_set)
# # for idx, k in enumerate(192):
# #     print("TestSample sample%d[]"%(idx))
# #     for i in range (40):
# #         for j in range(40):
# #             print("%d, "%(img[i][j]), end='')
# #         print()

# num_example = 75


# samples_file = open("test_samples.cc", "w")
# samples_file.write("#include \"test_samples.h\"\n\n")
# samples_file.write("const int kNumSamples = " + str(img_set.shape[0]) + ";\n\n")
# samples = "" 
# samples_array = "const TestSample test_samples[kNumSamples] = {"

# print(img_set.shape)
# for img_idx in range(img_set.shape[0]):
#     img_arr = list(np.ndarray.flatten(img_set[img_idx]))
#     print(np.array(img_arr).shape)
#     var_name = "sample" + str(img_idx)
#     samples += "TestSample " + var_name + " = {\n"
#     samples += "\t.label = " + "0" + ",\n" 
#     samples += "\t.image = {\n"
#     wrapped_arr = [img_arr[i:i + 20] for i in range(0, len(img_arr), 20)]
#     for sub_arr in wrapped_arr:
#         samples += "\t\t" + str(sub_arr)
#     samples += "\t}\n};\n\n"    
#     samples_array += var_name + ", "
#     samples = samples.replace("[", "")
# samples = samples.replace("]", ",\n")
# samples_array += "};\n"
# samples_file.write(samples)
# samples_file.write(samples_array)
# samples_file.close()

import numpy as np
from PIL import Image
from utils.magnet.worker import slice_data_for_arc

img = Image.open('dirty_chicken_Gray.jpg')
img = np.array(img)
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=0)
#print(img.shape)
img_set = slice_data_for_arc(img, 40, 40)
#print(img_set)
# for idx, k in enumerate(192):
#     print("TestSample sample%d[]"%(idx))
#     for i in range (40):
#         for j in range(40):
#             print("%d, "%(img[i][j]), end='')
#         print()

num_example = 10


samples_file = open("test_samples.cc", "w")
samples_file.write("#include \"test_samples.h\"\n\n")
samples_file.write("const int kNumSamples = " + str(num_example) + ";\n\n")
samples = "" 
samples_array = "const TestSample test_samples[kNumSamples] = {"

print(img_set.shape)
for img_idx in range(10):
    img_set = Image.open("./gray_data/dirty"+str(img_idx)+".png")
    img_set = np.array(img_set)
    img_arr = list(np.ndarray.flatten(img_set))
    print(np.array(img_arr).shape)
    var_name = "sample" + str(img_idx)
    samples += "TestSample " + var_name + " = {\n"
    samples += "\t.label = " + "0" + ",\n" 
    samples += "\t.image = {\n"
    wrapped_arr = [img_arr[i:i + 20] for i in range(0, len(img_arr), 20)]
    for sub_arr in wrapped_arr:
        samples += "\t\t" + str(sub_arr)
    samples += "\t}\n};\n\n"    
    samples_array += var_name + ", "
    samples = samples.replace("[", "")
samples = samples.replace("]", ",\n")
samples_array += "};\n"
samples_file.write(samples)
samples_file.write(samples_array)
samples_file.close()
