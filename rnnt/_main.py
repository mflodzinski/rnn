import rnnt.utils as utils
from rnnt.train import train_model

# torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    CONFIG_PATH = "config/config.yaml"

    config = utils.load_config(CONFIG_PATH)
    logger = utils.setup_logger(config)
    visualizer = utils.create_visualizer(config)
    device = utils.setup_device(logger)

    train_data, test_data, val_data, tokenizer = utils.prepare_data_loaders(
        config, logger
    )
    model = utils.initialize_model(config, tokenizer.vocab_size, device)
    optimizer = utils.create_optimizer(model, config.optim)
    utils.log_model_parameters(model, logger)

    train_model(
        config=config,
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        logger=logger,
        device=device,
        visualizer=visualizer,
        tokenizer=tokenizer,
    )
