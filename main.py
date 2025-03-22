from peft.cache_prompts import main as save_cache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_name = "updated_point_clouds.db"

def main():
    logger.info("Starting cache prompt saving...")
    save_cache(
        model_path="data/weights/pretrain.pth",
        data_path=f"data/{db_name}",
        output_path=f"data/cache_prompts",
        batch_size=16,
        num_workers=4,
        device="cuda"
    )
    logger.info("Cache prompt process completed...")

if __name__ == '__main__':
    main()