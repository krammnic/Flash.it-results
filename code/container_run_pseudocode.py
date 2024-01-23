from src.flash_formers.container.container import LLMContainer

container = LLMContainer(
    "ff474c7028321b8dafb9fcede8089d7b",
    "models/llama.cpp",
    k8s=False,
    inference="Marlin, FlashInference",
    quantization=True,
    end_point="common",
    include="infrastructure"
)

"""
Some description here is required. 

All containers are created locally. You need special API Token to run it.
Then you choose model and include your infrastructure.

Then you create container and magic and all optimizations happens.
"""

if __name__ == '__main__':
    container.set_up()
    container.test_run()
