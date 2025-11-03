from datasets import load_dataset
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix

def load_tahoe():
    """
    Utility to load the Tahoe-100M dataset
    """
    print("chargement tahoe-100m...")
    main_data = load_dataset("tahoebio/Tahoe-100M", 
                                 split="train", 
                                 streaming=True)
    gene_metadata = load_dataset("tahoebio/Tahoe-100M", 
                                 name="gene_metadata", 
                                 split="train")
    gene_vocab = {entry["token_id"]: entry["ensembl_id"] 
                    for entry in gene_metadata}
    return main_data, gene_vocab 

def create_anndata_from_generator(generator, 
                                  gene_vocab, 
                                  sample_size=None):
    """
    Inspired by the function given by the authors of the Tahoe-100M dataset to load properly the data
    see https://huggingface.co/datasets/tahoebio/Tahoe-100M/blob/main/tutorials/loading_data.ipynb
    """
    sorted_vocab_items = sorted(gene_vocab.items())
    token_ids, gene_names = zip(*sorted_vocab_items)
    token_id_to_col_idx = {token_id: idx 
                           for idx, token_id in enumerate(token_ids)}

    data, indices, indptr = [], [], [0]
    obs_data = []

    print("\nFetching samples")
    for i, cell in enumerate(generator.shuffle()):
        if sample_size is not None and i >= sample_size:
            break
        genes = cell['genes']
        expressions = cell['expressions']
        if expressions[0] < 0: 
            genes = genes[1:]
            expressions = expressions[1:]

        col_indices = [token_id_to_col_idx[gene] for gene in genes if gene in token_id_to_col_idx]
        valid_expressions = [expr for gene, expr in zip(genes, expressions) if gene in token_id_to_col_idx]

        data.extend(valid_expressions)
        indices.extend(col_indices)
        indptr.append(len(data))

        obs_entry = {k: v for k, v in cell.items() if k not in ['genes', 'expressions']}
        obs_data.append(obs_entry)

    print("Creating AnnData..")
    expr_matrix = csr_matrix((data, indices, indptr), 
                             shape=(len(indptr) - 1, len(gene_names)))
    obs_df = pd.DataFrame(obs_data)

    adata = ad.AnnData(X=expr_matrix, obs=obs_df)
    adata.var.index = pd.Index(gene_names, name='ensembl_id')

    return adata

def subsample_tahoe(dataset_tahoe, 
                    gene_vocab,
                    size, 
                    n_genes: int = 500,
                    gene_selection: str = 'hvg'):
    """
    Subsample the dataset and applies preprocessing
    """
    sample_data = create_anndata_from_generator(dataset_tahoe, gene_vocab, size)

    # 1) Sequencing depth normalization
    print("\nSequencing depth normalization..")
    sc.pp.normalize_total(sample_data, inplace=True)
    print(sample_data)
    # 2) gene selection
    print("Gene selection..")
    if gene_selection == "first":
        sample_data = sample_data[:n_genes]  # premiers 500 gènes
    else:
        # Keep n_genes Highly Variable Genes (HGV)
        sc.pp.highly_variable_genes(sample_data, 
                                    inplace=True,
                                    n_top_genes=n_genes)
        sample_data = sample_data[:,sample_data.var['highly_variable'] == True]
    
    print("Log normalization..")
    # 3) switch to log1p domain
    sc.pp.log1p(sample_data) # inplace
    sample_labels = list(sample_data.obs['drug'])
    sample_data = np.array(sample_data.X.todense())

    print(f"\nforme données: {sample_data.shape}")
    return sample_data, sample_labels

def prepare_tahoe_dataset(size, n_genes=500, gene_selection='hvg'):
    main_data, gene_vocab = load_tahoe()
    return subsample_tahoe(main_data, gene_vocab, size, 
                           n_genes, gene_selection)
