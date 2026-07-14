"""Analyses that produce the paper's figures and tables.

Each module is runnable on its own (``python -m capara.analysis.<module> --help``) and
writes to ``results/figures``, ``results/tables`` and ``results/<analysis>`` by default.

Figures and tables
    ``results_tables``   -- the two retrieval tables
    ``pareto``           -- short-caption vs long-description trade-off
    ``training_dynamics``-- loss and validation recall per epoch
    ``text_length``      -- retrieval as a function of the text truncation length
    ``embedding_space``  -- t-SNE and image-text alignment, per dataset
    ``cross_dataset_alignment`` -- alignment across all four datasets
    ``tsne_grid``        -- 4x3 t-SNE grid (datasets x models)
    ``qualitative``      -- retrieval examples where models disagree
    ``saliency``         -- gradient saliency maps
    ``dataset_stats``    -- statistics of the CC3M-derived training set
    ``dataset_quality``  -- GPT-4o LLM-as-a-judge scoring of the generated texts

Shared helpers
    ``style``   -- matplotlib rcParams, palettes, config labels and ordering
    ``samples`` -- benchmark image-text pair loading
    ``models``  -- CLIP and Long-CLIP wrappers, ranking helpers (BLIP lives in
    ``capara.common.blip``)
"""
