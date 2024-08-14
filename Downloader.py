import gzip
gz_file = 'UrbanSound8K.tar.gz'
with gzip.open(gz_file, 'rt') as f:
    data = f.read()
print(data[:500])
print("done")