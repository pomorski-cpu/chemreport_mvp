The runtime read-across dataset is stored here as a single SQLite file:
- `read_across.db`

Optional raw source files can also live here while rebuilding the database:
- `dataset_to_article.csv`
- `logPwithSmiles.csv`
- `genotoxicity_binary.csv`

Raw CSV files are ignored by Git. The SQLite database can be committed and used by the app directly.
