import mindspore as ms
import torch

def convert_torch2ms(torch_ckpt, ms_ckpt, new_ms_ckpt='new.ckpt'):
    torch_dict = torch.load(torch_ckpt)['model'].state_dict()
    ms_dict = ms.load_checkpoint(ms_ckpt)
    param_list = []

    for t_k in torch_dict.keys():
        if 'num_batches_tracked' in t_k:
            continue

        new_k = 'model.' + t_k[:]
        if '.bn.' in new_k:
            if 'running_mean' in new_k:
                new_k = new_k[:-len('running_mean')] + 'moving_mean'
            elif 'running_var' in new_k:
                new_k = new_k[:-len('running_var')] + 'moving_variance'
            elif 'weight' in new_k:
                new_k = new_k[:-len('weight')] + 'gamma'
            elif 'bias' in new_k:
                new_k = new_k[:-len('bias')] + 'beta'


        replace_k = ['model.model.2.m', 'model.model.4.m', 'model.model.6.m',
                     'model.model.8.m', 'model.model.9', 'model.model.12.m',
                     'model.model.15.m', 'model.model.18.m', 'model.model.21.m']

        for r_k in replace_k:
            if new_k.startswith(r_k):
                if 'cv1' in new_k:
                    index = new_k.find('cv1')
                    new_k = new_k[:index] + 'conv1' + new_k[index+len('cv1'):]
                elif 'cv2' in new_k:
                    index = new_k.find('cv2')
                    new_k = new_k[:index] + 'conv2' + new_k[index + len('cv2'):]
                else:
                    print("cv1/cv2 not in new_k.")
                break

        assert new_k in ms_dict, f"Not match Key, torch: {t_k}, new_k: {new_k}."
        param_list.append({"name": new_k, "data": ms.Tensor(torch_dict[t_k].numpy())})
    ms.save_checkpoint(param_list, new_ms_ckpt)


if __name__ == '__main__':
    torch_ckpt = "yolov8x-seg.pt"
    ms_ckpt = "yolov8x-seg-1_924.ckpt"
    new_ms_ckpt = "yolov8x_seg_from_torch.ckpt"
    convert_torch2ms(torch_ckpt, ms_ckpt, new_ms_ckpt)
