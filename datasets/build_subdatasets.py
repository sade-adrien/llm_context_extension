from datasets import load_dataset
from huggingface_hub import notebook_login
import os

huggingface_api_key = "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"
os.environ["HUGGINGFACE_TOKEN"] = huggingface_api_key
notebook_login()
os.environ.pop("HUGGINGFACE_TOKEN", None)

dataset = load_dataset("togethercomputer/RedPajama-Data-V2",
                  name="default",
                  partition="head_middle",
                  snapshots=["2023-14"],
                  languages=["en"],
                  cache_dir="/mnt/esperanto/et/Adrien/context_extension/dataset",
                  data_dir='/mnt/esperanto/et/Adrien/context_extension/dataset',
                  )

subdataset2 = dataset['train'].select(list(range(1_000_000, 11_000_000)))
subdataset2.push_to_hub(repo_id='redpajama_v2_sample_10M', private=False, max_shard_size="2GB")

subdataset3 = dataset['train'].select(list(range(11_000_000, 111_000_000)))
subdataset3.push_to_hub(repo_id='redpajama_v2_sample_100M', private=False, max_shard_size="8GB")

