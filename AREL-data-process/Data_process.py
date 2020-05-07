import json
import os.path as osp
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter
import numpy
import h5py
import os


"""
处理文本数据，提取出story，并构建词表
"""
base_path = "AREL-data-process/"
train_data = json.load(open(osp.join(base_path, "train.story-in-sequence.json")))
val_data = json.load(open(osp.join(base_path, "val.story-in-sequence.json")))
test_data = json.load(open(osp.join(base_path, "test.story-in-sequence.json")))

### 处理图像数据
prefix = ["train", "val", "test"]
whole_album2im = {}
for i, data in enumerate([train_data, val_data, test_data]):
    album2im = {} # 按照album存储图像数据，键为album_id，值为img_id，1-多
    for im in data['images']: # 遍历每一张图像
        if im['album_id'] not in album2im: # 以album区分，若album_id并未存储，则为新的album
            album2im[im['album_id']] = [im['id']]
        else: # 该album已存在，则append，注意数据已经按照时间排序
            if im['id'] not in album2im[im['album_id']]:
                album2im[im['album_id']].append(im['id'])
    whole_album2im[prefix[i]] = album2im

for i, data in enumerate([train_data, val_data, test_data]):
    a = [] # 按照album存储图像数据，键为album_id，值为img_id，1-多
    for im in data['images']: # 遍历每一张图像
        if im['id'] not in a: # 以album区分，若album_id并未存储，则为新的album
            a.append(im['id'])
    print(len(a))

### 处理文本数据
whole_album = {}
story_lines = {} # 存储每个故事，每个故事五句话，index为0、5、10
whole_lines = {} # 存储每个故事，一行存储，index为0、1、2
story_line_count = 0 # 句子数量
whole_line_count = 0 # story数量
for i, data in enumerate([train_data, val_data, test_data]):
    album_mapping = {} # 存储story
    for annot_new in data["annotations"]: # 遍历每组数据
        annot = annot_new[0] # album_id
        assert len(annot_new) == 1
        text = bytes.decode(annot['text'].encode('utf8')) # 字段中包含origin_text和text，前者为原始文本，后者为匿名文本
        if annot['story_id'] not in album_mapping: # story_id为这段描述的id(5张图片)
            album_mapping[annot['story_id']] = {"text_index": [story_line_count], "flickr_id": [annot['photo_flickr_id']], "length": 1, 
                                                "album_id": annot['album_id'], "album_flickr_id": whole_album2im[prefix[i]][annot['album_id']],
                                                "whole_text_index": whole_line_count, "origin_text": text} # story_line_count表示一个句子的id，flickr_id为图片id，album_id，album_flickr_id对应album对应的图像列表，whole_text_index<story_line_count即为story数量
            story_lines[annot['story_id']] = [{"index": story_line_count, "text": text.split()}]
            whole_lines[annot['story_id']] = {"index": whole_line_count, "text": text.split()}
            whole_line_count +=1
        else:
            album_mapping[annot['story_id']]["text_index"].append(story_line_count)
            album_mapping[annot['story_id']]["flickr_id"].append(annot['photo_flickr_id'])
            album_mapping[annot['story_id']]["length"] += 1 # length计算当前story长度，1-5
            story_lines[annot['story_id']].append({"index": story_line_count, "text": text.split()}) 
            whole_lines[annot['story_id']]["text"].extend(text.split())
            album_mapping[annot['story_id']]["origin_text"] += " " + text
        story_line_count += 1
    whole_album[prefix[i]] = album_mapping

new_story_lines = [] 
for l in story_lines.values():
    for li in l:
        new_story_lines.append(li)
story_lines = new_story_lines
whole_lines = whole_lines.values()

story_lines = [r['text'] for r in sorted(story_lines, key=lambda thing: thing['index'])] # 一个句子存储一行
whole_lines = [r['text'] for r in sorted(whole_lines, key=lambda thing: thing['index'])] # 一个故事存储一行

print(len(story_lines))
print(len(whole_lines))


cnt = Counter() # 可以进行计数，词表构建（词：出现次数）
for l in story_lines:
    words = l
    for w in words:
        cnt[w] += 1
words2id = {}
idx = 2
## 构建词表
for k, v in cnt.most_common():
    if v > 5:
        words2id[k] = idx
        idx += 1
words2id["<EOS>"] = 0
words2id["<UNK>"] = 1
id2words = {v:k for k,v in words2id.items()}
print(len(id2words))

whole_album["words2id"] = words2id
whole_album["id2words"] = {v:k for k,v in words2id.items()}

# 将文本转换为id
id_story_lines = []
for l in story_lines:
    s = [words2id[w] if w in words2id else 1 for w in l]
    id_story_lines.append(s)

id_whole_lines = []
for l in whole_lines:
    s = [words2id[w] if w in words2id else 1 for w in l]
    id_whole_lines.append(s)

# 进行padding，padding为0，长度为105
new_id_whole_lines = []
specify_longest = 105
for i in range(len(id_whole_lines)):
    cur_len = len(id_whole_lines[i])
    if cur_len < specify_longest:
        new_id_whole_lines.append(id_whole_lines[i] + [0] * (specify_longest - cur_len))
    else:
        new_id_whole_lines.append(id_whole_lines[i][:specify_longest-1] + [0])
# shape(50200,105)
data = numpy.asarray(new_id_whole_lines)

# f = h5py.File("full_story.h5", "w")
# f.create_dataset("story", data=data)
# f.close()
## 对单个句子进行padding，大小为30
new_id_story_lines = []
specify_longest = 30
for i in range(len(id_story_lines)):
    cur_len = len(id_story_lines[i])
    if cur_len < specify_longest:
        new_id_story_lines.append(id_story_lines[i] + [0] * (specify_longest - cur_len))
    else:
        new_id_story_lines.append(id_story_lines[i][:specify_longest-1] + [0])
## （25100，30）
data = numpy.asarray(new_id_story_lines, "int32")

# f = h5py.File("story.h5", "w")
# f.create_dataset("story", data=data)
# f.close()

# # 删除图像少于5张的
# for p in prefix:
#     path = "/mnt/sshd/wenhuchen/VIST/images_256/{}/".format(p)
#     deletables = []
#     for story_id, story in whole_album[p].items():
#         d = [osp.exists(osp.join(path, "{}.jpg".format(_))) for _ in story["flickr_id"]]
#         if sum(d) < 5:
#             print("deleting {}".format(story_id))
#             deletables.append(story_id)
#         else:
#             pass
#     for i in deletables:
#         del whole_album[p][i]

# 构建图像与story的映射
flickr_story_map = {}
for pre in prefix:
    album = whole_album[pre]
    for k, v in album.items():
        indexes = v['text_index']
        for i, flickr_id in enumerate(v['flickr_id']):
            if flickr_id not in flickr_story_map:
                flickr_story_map[flickr_id] = [indexes[i]]
            else:
                flickr_story_map[flickr_id].append(indexes[i])

# 画出story的长度分布
# length_distribution = [len(s) for s in whole_lines]
# result = plt.hist(length_distribution, bins='auto', cumulative=True, normed=1)
# plt.show()
# length_distribution = [len(s) for s in story_lines]
# result = plt.hist(length_distribution, bins='auto', cumulative=True, normed=1)
# plt.hist(length_distribution, bins='auto')
# plt.show()


"""
处理文本数据，提取出caption
"""
base_path = "AREL-data-process/dii/"
train_data = json.load(open(osp.join(base_path, "train.description-in-isolation.json")))
val_data = json.load(open(osp.join(base_path, "val.description-in-isolation.json")))
test_data = json.load(open(osp.join(base_path, "test.description-in-isolation.json")))

mapping = {}
mapping_original = {}
text_list = []
text_list_count = 0
unknown_words = 0
total_words = 0
with_story = 0
no_story = 0
for i, data in enumerate([train_data, val_data, test_data]):
    mapping[prefix[i]] = {}
    mapping_original[prefix[i]] = {}
    for l in data['annotations']:
        if l[0]['photo_flickr_id'] not in mapping[prefix[i]]:
            if l[0]['photo_flickr_id'] in flickr_story_map:
                stories =  flickr_story_map[l[0]['photo_flickr_id']]
            else:
                stories = [-1]
            mapping[prefix[i]][l[0]['photo_flickr_id']] = {'caption': [text_list_count], 'story': stories}
            mapping_original[prefix[i]][l[0]['photo_flickr_id']] = [l[0]['text']]
        else:
            mapping[prefix[i]][l[0]['photo_flickr_id']]['caption'].append(text_list_count)
            mapping_original[prefix[i]][l[0]['photo_flickr_id']].append(l[0]['text'])
        text_list_count += 1
        assert len(l) == 1
        s = []
        for w in l[0]['text'].split(" "):
            if w in words2id:
                s.append(words2id[w])  
            else:
                s.append(1)
                unknown_words += 1
            total_words += 1
        text_list.append(s)
for pre in prefix:
    count = 0
    for i in mapping[pre]:
        value = mapping[pre][i]
        if len(value['caption']) == 0:
            count += 1
    print(count)

print("unknown words percent is {}".format(unknown_words / (total_words + 0.0)))
new_text_list = []
specify_longest = 20
for i in range(len(text_list)):
    cur_len = len(text_list[i])
    if cur_len < specify_longest:
        new_text_list.append(text_list[i] + [0] * (specify_longest - cur_len))
    else:
        new_text_list.append(text_list[i][:specify_longest - 1] + [0]) 

# for p in prefix:
#     path = "/mnt/sshd/wenhuchen/VIST/images_256/{}/".format(p)
#     deletables = []
#     for flickr_id, story in mapping[p].items():
#         if not osp.exists(osp.join(path, "{}.jpg".format(flickr_id))):
#             deletables.append(flickr_id)
#     for i in deletables:
#         del mapping[p][i]
#         del mapping_original[p][i]
        
whole_album["image2caption"] = mapping
whole_album["image2caption_original"] = mapping_original

# with open("story_line.json", 'w') as f:
#     json.dump(whole_album, f)

text_array = numpy.asarray(new_text_list, dtype='int32')

# f = h5py.File("description.h5", 'w')
# f.create_dataset("story", data=text_array)
# f.close()

val_data = json.load(open(osp.join(base_path, "val.description-in-isolation.json")))
with open("val_desc_reference", "w") as f:
    for l in val_data['annotations']:
        # print >> f, "{}\t{}".format(l[0]['photo_flickr_id'], l[0]['text'])
        print(l[0]['photo_flickr_id'], l[0]['text'])

f = h5py.File("full_story.h5", "r")
print(f['story'][0])

f = h5py.File("story.h5", "r")
print(f['story'].shape)

f = open("story_line.json", 'r')
data = json.load(f)
print(len(data['id2words']))

# zero_fc = numpy.zeros((2048, ), "float32")
# zero_conv = numpy.zeros((2048, 7, 7), "float32")

# train_fc_base = "/mnt/sshd/xwang/VIST/feature/train/fc"
# train_conv_base = "/mnt/sshd/xwang/VIST/feature/train/conv"
# train_name1 = [l.split(".")[0] for l in os.listdir(train_fc_base)]

# train_image_base = "/mnt/sshd/wenhuchen/VIST/images/train"
# train_name2 = [l.split(".")[0] for l in os.listdir(train_image_base)]

# rest = set(train_name2) - set(train_name1)
# for image in rest:
#     numpy.save(os.path.join(train_fc_base, "{}.npy".format(image)), zero_fc) 
#     numpy.save(os.path.join(train_conv_base, "{}.npy".format(image)), zero_conv) 

# val_fc_base = "/mnt/sshd/xwang/VIST/feature/val/fc"
# val_conv_base = "/mnt/sshd/xwang/VIST/feature/val/conv"
# val_name1 = [l.split(".")[0] for l in os.listdir(val_fc_base)]

# val_image_base = "/mnt/sshd/wenhuchen/VIST/images/val"
# val_name2 = [l.split(".")[0] for l in os.listdir(val_image_base)]

# rest = set(val_name2) - set(val_name1)
# for image in rest:
#     numpy.save(os.path.join(val_fc_base, "{}.npy".format(image)), zero_fc) 
#     numpy.save(os.path.join(val_conv_base, "{}.npy".format(image)), zero_conv) 

# test_fc_base = "/mnt/sshd/xwang/VIST/feature/test/fc"
# test_conv_base = "/mnt/sshd/xwang/VIST/feature/test/conv"
# test_name1 = [l.split(".")[0] for l in os.listdir(test_fc_base)]

# test_image_base = "/mnt/sshd/wenhuchen/VIST/images/test"
# test_name2 = [l.split(".")[0] for l in os.listdir(test_image_base)]

# rest = set(test_name2) - set(test_name1)
# for image in rest:
#     numpy.save(os.path.join(test_fc_base, "{}.npy".format(image)), zero_fc) 
#     numpy.save(os.path.join(test_conv_base, "{}.npy".format(image)), zero_conv) 

# with open("story_line.json", 'r') as f: 
#     data = json.load(f)

# print(len(data['image2caption']['train']))
# print(len(data['train']))