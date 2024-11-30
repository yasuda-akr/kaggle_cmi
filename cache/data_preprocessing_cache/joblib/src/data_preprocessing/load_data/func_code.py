# first line: 19
@memory.cache
def load_data(data_dir):
    train = pl.read_csv(os.path.join(data_dir, 'train.csv')).to_pandas()
    test = pl.read_csv(os.path.join(data_dir, 'test.csv')).to_pandas()
    sample_submission = pl.read_csv(os.path.join(data_dir, 'sample_submission.csv')).to_pandas()
    wandb.log({'train_shape': train.shape, 'test_shape': test.shape})
    return train, test, sample_submission
