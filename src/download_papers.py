import os
import urllib.request

PAPERS = [
    ("https://arxiv.org/pdf/1706.03762", "attention_is_all_you_need.pdf"),
    ("https://arxiv.org/pdf/2005.14165", "gpt3.pdf"),
    ("https://arxiv.org/pdf/1810.04805", "bert.pdf"),
    ("https://arxiv.org/pdf/2106.09685", "lora.pdf"),
    ("https://arxiv.org/pdf/2006.04768", "longformer.pdf"),
    ("https://arxiv.org/pdf/2001.08361", "reformer.pdf"),
    ("https://arxiv.org/pdf/2009.14794", "performer.pdf"),
    ("https://arxiv.org/pdf/1910.10683", "t5.pdf"),
    ("https://arxiv.org/pdf/1906.08237", "xlnet.pdf"),
    ("https://arxiv.org/pdf/1907.11692", "roberta.pdf"),
    ("https://arxiv.org/pdf/2103.00020", "chinchilla.pdf"),
    ("https://arxiv.org/pdf/2010.11929", "vision_transformer.pdf"),
    ("https://arxiv.org/pdf/1509.02971", "ddpg.pdf"),
    ("https://arxiv.org/pdf/1707.06347", "ppo.pdf"),
    ("https://arxiv.org/pdf/1802.09477", "rainbow.pdf"),
    ("https://arxiv.org/pdf/1812.05905", "sac.pdf"),
    ("https://arxiv.org/pdf/1312.5602", "dqn.pdf"),
    ("https://arxiv.org/pdf/2106.01345", "decision_transformer.pdf"),
]

def download():
    os.makedirs("data/raw", exist_ok=True)
    for url, name in PAPERS:
        path = f"data/raw/{name}"
        if not os.path.exists(path):
            print("Downloading", name)
            urllib.request.urlretrieve(url, path)

if __name__ == "__main__":
    download()