# first line: 29
@memory.cache
def process_file(filename, dirname):
    df = pl.read_parquet(os.path.join(dirname, filename, 'part-0.parquet')).to_pandas()
    df.drop('step', axis=1, inplace=True)
    if np.any(np.isinf(df)):
        df = df.replace([np.inf, -np.inf], np.nan)
    return df.describe().values.reshape(-1), filename.split('=')[1]
