# Na-ion Cathode Discovery: High-Throughput ML Screening

A comprehensive computational search for novel sodium-ion battery cathode materials, combining high-throughput screening of Materials Project data, GNN-based voltage prediction with transfer learning from Li-ion, and ML universal potential validation using M3GNet/CHGNet.

## Motivation

Sodium-ion batteries are emerging as a commercial alternative to lithium-ion for grid storage and low-cost EVs, with CATL and BYD already shipping Na-ion cells. However, the cathode design space for Na-ion is far less explored than Li-ion. This project applies a modern computational funnel (database mining, ML prediction, ML potential validation) to identify promising Na-ion cathode candidates for future DFT and experimental study.

## Pipeline

1. **Data collection**: Mine Materials Project for 3,000+ Na-containing structures and 500+ known Na insertion electrodes
2. **GNN voltage predictor**: Train CGCNN on Na-ion voltage data with transfer learning from Li-ion
3. **High-throughput screening**: Predict voltage and capacity for all candidates; filter by stability, voltage, and capacity
4. **ML potential validation**: Relax top candidates with M3GNet/CHGNet; estimate Na migration barriers via NEB
5. **Benchmarking**: Compare discovered candidates against established Na-ion cathodes
6. **Dashboard**: Interactive Plotly visualizations and sortable candidate tables

## Key Results

*Results will be populated after running notebooks 01 through 06.*

## Repository Layout

```
na-ion-cathode-discovery/
  README.md
  LICENSE
  environment.yml
  notebooks/
    01_data_collection.ipynb
    02_gnn_voltage_predictor.ipynb
    03_screening.ipynb
    04_ml_potential_validation.ipynb
    05_benchmarking.ipynb
    06_dashboard.ipynb
  src/
    __init__.py
    data.py
    models.py
    train.py
    evaluate.py
    screen.py
    validate.py
    utils.py
  data/
  models/
  results/
```

## Quick Start

### 1. Create environment

```bash
conda env create -f environment.yml
conda activate na-cathode
python -m ipykernel install --user --name na-cathode --display-name "na-cathode"
```

### 2. Set Materials Project API key

```bash
export MP_API_KEY="your_key_here"
```

### 3. Run notebooks in order

```bash
jupyter lab
```

Run notebooks 01 through 06 sequentially.

## Methodology Notes

This project uses ML universal potentials (M3GNet, CHGNet) in place of DFT calculations. These potentials approximate DFT accuracy at a fraction of the computational cost and run on consumer GPUs. All ML-potential results should be validated with DFT before experimental synthesis decisions. This "computational funnel" approach (database screening, ML prediction, ML-potential validation, DFT recommendation) reflects how modern materials discovery pipelines operate at national laboratories and in industry.

Transfer learning from Li-ion to Na-ion voltage prediction is a published and validated approach. The larger Li-ion training set provides a structural prior that improves prediction accuracy on the smaller Na-ion dataset.

## Citation

If you use this project, please cite the Materials Project:

Jain, A. et al. (2013). The Materials Project: A materials genome approach to accelerating materials innovation. *APL Materials*, 1, 011002.

## License

MIT
