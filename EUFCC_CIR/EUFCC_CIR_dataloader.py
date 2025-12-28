from EUFCC_CIR_downloader import download_images
import pandas as pd

df = pd.read_csv('test_ood.csv')
download_images(df, root_dir='data/test_ood')