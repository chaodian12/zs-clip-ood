# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import tqdm
import fire
import json

from llama import Llama
from tqdm import tqdm
import torch.distributed as dist
#dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29501', world_size=2, rank=0)
from collections import Counter

def check_duplicate_words(input_string, threshold=80):
    # 将输入字符串拆分为单词列表
    words = input_string.split()
    # 使用Counter统计每个单词的出现次数
    word_counts = Counter(words)
    # 检查是否有单词重复超过阈值
    for word, count in word_counts.items():
        if count > threshold:
            return True
    return False

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.9,
    top_p: float = 0.2,
    max_seq_len: int = 128,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    #/media/chaod/code/TaI-DPT/datasets/llm/imagenet1kood16.json
    with open('/media/chaod/code/TaI-DPT/datasets/llm/imagenet1kood16.json', 'r') as f:
        datas = json.load(f)
    
    print(len(datas))
    # 逐步写入数据到文件
    # for _ in tqdm(range(0,100)):
    for i in range(0, len(datas), max_batch_size):
        batch_datas = datas[i:i+max_batch_size]
        if check_duplicate_words(datas[i]["content"]) == True:
            continue
        dialogs_list = [[{"role": "user", "content": data["content"]}] for data in batch_datas]

        # 逐步写入数据到文件
        #! 1 cifar10_id_answ
        #! 2 cifar10_oodsam_answ
        #! 4 cifar10_oid_answ
        with open('/media/chaod/code/TaI-DPT/datasets/llm/imagenet1kood16 _answ.json', 'a') as file:
            results_list = generator.chat_completion(
                dialogs_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            for data, results in zip(batch_datas, results_list):
                data["content"] = results['generation']['content']
                json.dump(data, file)
                file.write("\n") 

if __name__ == "__main__":
    
    fire.Fire(main)
