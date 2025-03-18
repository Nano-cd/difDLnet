# # 在代码开始处添加这段代码，检查Java是否可用
# import subprocess
#
# cmd = ['java', '-cp', "E:/project_pycharm/difDLnet/evaluation/stanford-corenlp-3.4.1.jar", \
#            'edu.stanford.nlp.process.PTBTokenizer', \
#            '-preserveLines', '-lowerCase']
# subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT)
# print("Java已安装")
# # except:
# #     print("找不到Java，请安装Java或设置正确的PATH环境变量")


import h5py
with h5py.File("D:/BaiduNetdiskDownload/aaai21_DLCT/coco_all_align/coco_all_align.hdf5", 'r') as f:
    print(f.keys())  # 应该包含'detections'等key
