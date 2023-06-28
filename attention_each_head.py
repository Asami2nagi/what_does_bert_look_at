import torch
import yaml

from transformers import BertJapaneseTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer
from transformers import MLukeTokenizer, LukeModel
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib

import os

__all__ = ["japanize_matplotlib"]


def main(config):




    # モデルの読み込み
    model_name_list = config["model"]["name"]
    for model_name in model_name_list:
        print(f"Now Loading ... : {model_name}")
        if model_name == "cl-tohoku/bert-base-japanese":
            tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
            # output_attentions=TrueでAttentionを取得できる
            model = BertModel.from_pretrained(model_name, output_attentions=True)
            gen_attention(tokenizer,model,model_name)
        elif model_name == "rinna/japanese-roberta-base":
            # load tokenizer
            tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-roberta-base")
            tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
            # load model
            model = RobertaForMaskedLM.from_pretrained(model_name, output_attentions=True)
            gen_attention(tokenizer,model,model_name)
        elif model_name == "rinna/japanese-gpt2-medium":
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
            model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
            gen_attention(tokenizer,model,model_name, nrows=4)
        elif model_name == "studio-ousia/luke-japanese-base":
            tokenizer = MLukeTokenizer.from_pretrained(model_name)
            model = LukeModel.from_pretrained(model_name, output_attentions=True )
            gen_attention(tokenizer,model,model_name)
        elif model_name == 'cyberagent/open-calm-7b':
            #model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-7b", device_map="auto", torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
            gen_attention(tokenizer, model, model_name, nrows=8)

# もしAttentionHeadの数が12以上ある場合は、nrowsを変更する
# 与える値はHead数を4で割った値(切り上げ)

def gen_attention(tokenizer,model,model_name, nrows=4):
     # テキストの読み込み
    with open("./src/sentence.txt", encoding="utf-8") as f:
        index = 1
        # 一行ずつ読み込み while文で回す
        # readline()は最後に改行文字が含まれる
        while True:
            text = f.readline()
            if text == "":
                break

            print("------------------------")
            print(text)

            tokens = tokenizer.encode_plus(
                text,
                return_tensors="pt",
                add_special_tokens=True,
                max_length=514,
                truncation=True,
            )  # 文章をトークン化
            # Attentionの取得
            outputs = model(
                tokens["input_ids"], attention_mask=tokens["attention_mask"]
            )  # type: ignore # Attentionの取得
            gen_picture(tokens, outputs, tokenizer, model_name, index, nrows=nrows)
            index+=1

def gen_picture(tokens, outputs, tokenizer, model_name, index, nrows):
    # ヒートマップの描画
    for i, row_attention in enumerate(outputs.attentions):
        fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(12, 9))
        print("Now Printing ... : Attention", i + 1)
        for j, head_attention in enumerate(row_attention[0]):
            print("Now Printing ... : Attention", i + 1, " /", "head", j + 1)
            attention = head_attention.detach().numpy()
            print(tokenizer.decode(tokens["input_ids"][0]))
            #from sample code
            #output = tokenizer.decode(tokens[0], skip_special_tokens=True)
            sns.heatmap(
                attention,
                cmap="YlGnBu",
                #xticklabels=tokenizer.convert_ids_to_tokens( tokens["input_ids"][0]
                xticklabels=tokenizer.decode(

                ),
                #yticklabels=tokenizer.convert_ids_to_tokens(
                yticklabels=tokenizer.decode(
                        tokens["input_ids"][0]
                ),
                ax=axes[j // 4][j % 4],
                #ax=axes[j // 8][(j % 8)//4], #j // 8j%4 to(j % 4)/2
            )
            axes[j // 4][j % 4].set_title(f"Attention {i+1} / head {j+1}")
            #axes[j // 8][(j % 8)//4].set_title(f"Attention {i+1} / head {j+1}")
            plt.tight_layout()
            # 保存
        # フォルダがなければ作成
        if not os.path.exists(f"./fig/output_each_model/{model_name}"):
            os.makedirs(f"./fig/output_each_model/{model_name}")
        if not os.path.exists(f"./fig/output_each_model/{model_name}/{index}"):
            os.makedirs(f"./fig/output_each_model/{model_name}/{index}")
        plt.savefig(
            f"./fig/output_each_model/{model_name}/{index}/attention_layer_{i+1}_head_{j+1}.png"
        )

        # 前のグラフをクリア
        plt.clf()
        plt.close()


if __name__ == "__main__":
    # 設定ファイルの読み込み
    with open("config.yaml", encoding="utf-8") as yml:
        config = yaml.safe_load(yml)
    main(config)
