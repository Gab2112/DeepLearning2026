# DeepLearning2026 — Forecasting Vegetation Health from Satellite Time Series

Il progetto usa tre osservazioni satellitari `clear-sky`, meteo giornaliero storico e futuro perturbato, informazioni sulle nubi, calendario e clima per prevedere NDVI, GNDVI ed EVI alle tre acquisizioni successive. L'obiettivo è stimare i quantili 0.1, 0.5 e 0.9, per un totale di 27 valori per finestra. Il repository contiene il loader, gli imputatori storici, le baseline e le metriche. Il forecaster neurale probabilistico è ancora da implementare. [Traccia del progetto](DL_Project.pdf) · [Documentazione del dataset](DATA/README.md)

## Imputazione degli indici storici

`interpolation.strategy: "linear"` è l'impostazione predefinita per NDVI, GNDVI ed EVI storici nelle colonne 18–20. La factory comprende anche Bi-GRU (`bigru` o `model`), MAE Transformer (`mae`), spline e altri metodi. La scelta dell'imputatore e il profilo meteorologico sono due opzioni indipendenti. Il benchmark leave-one-out misura la ricostruzione storica. Il contributo al forecasting finale va misurato con una valutazione dedicata.

## Profili meteorologici

Impostare `weather_preprocessing.profile` in [configs/config.yaml](configs/config.yaml).

| Profilo | Meteo storico nel loader di forecasting | Cleaning storico | Meteo futuro |
|---|---|---|---|
| M0 (default) | Interpolazione lineare, Arcsinh, zero filling residuo | Disattivo | Rumore originale |
| M1 | Arcsinh, zero filling normalizzato e otto maschere | Disattivo | Rumore originale |
| M2 | Come M1 | Attivo | Rumore originale |
| M3 | Come M1 | Attivo | Clipping opzionale sulle covariate perturbate |

`src/data/weather_preprocessor.py` lavora su copie delle 18 covariate. In M0 il loader usa l'interpolatore lineare senza clipping e ne mantiene il fallback a zero grezzo per le colonne interamente mancanti. Per i vecchi imputatori neurali, M0 conserva la normalizzazione e lo zero filling storico senza interpolare il meteo. Le statistiche Arcsinh provengono da `DATA/train_multi_output_clear_sky/scaler.json` per ogni profilo.

Da M1, una covariata è disponibile quando è finita e la maschera sorgente la indica come osservata. I valori indisponibili ricevono zero nello spazio normalizzato. Il modello può distinguere questo placeholder da una misura pari alla media del train con il flag meteorologico individuale. I flag valgono 1 per una misura utilizzabile e 0 per una misura indisponibile.

Il dataset espone tre maschere storiche a otto colonne. `history_weather_original_missing_mask` unisce i NaN e la maschera sorgente. `history_weather_invalid_mask` registra le anomalie aggiunte dal cleaning. `history_weather_mask` indica la disponibilità effettiva. Per il futuro espone `future_weather_original_missing_mask` e `future_weather_mask`. Le maschere meteorologiche, la maschera delle osservazioni vegetative e le padding mask sono tensori distinti.

### Cleaning storico e clipping futuro

La scheda E-OBS indica vento in m/s, umidità relativa in percentuale, irraggiamento in W/m², precipitazioni giornaliere in mm, pressione a livello del mare in hPa e temperature in °C. Il cleaning storico segnala valori non finiti ulteriori rispetto ai NaN originali, vento, pioggia o irraggiamento negativi, umidità fuori da [0, 100] e pressione non positiva. Gli estremi statistici finiti entro queste regole restano disponibili. [Unità delle variabili E-OBS](https://cds.climate.copernicus.eu/datasets/insitu-gridded-observations-europe?tab=overview)

Per le temperature il controllo confronta i tre possibili ordini tra coppie disponibili. `temperature_tolerance_c: 1.0` è una soglia sperimentale. Con `temperature_policy: "mask_triplet"` un'inversione che supera la tolleranza invalida tutti i membri ancora disponibili della terna. `temperature_policy: "ignore"` disattiva il solo controllo termico. I valori invalidati passano a NaN nella copia elaborata prima della normalizzazione.

In M0, M1 e M2 il futuro usa direttamente i valori perturbati finiti. M3 porta a zero vento, pioggia e irraggiamento negativi e limita umidità e copertura nuvolosa a [0, 100]. Per le nubi la scala percentuale è inferita dalle frazioni documentate e dai valori storici del train, che arrivano fino a 100. M3 conserva i valori perturbati di pressione e temperatura e la disponibilità dei valori sottoposti a clipping. Il loader usa `future` e `future_mask` ed esclude `future_noise` e le colonne vegetative future. I target indisponibili restano NaN e sono segnalati da `target_mask`.

## Dimensioni di input e checkpoint

| Componente | M0 | M1, M2, M3 |
|---|---:|---:|
| `VegetationDataset` storico | 21 | 29 |
| `VegetationDataset` futuro | 18 | 26 |
| Input storico degli imputatori neurali | 27 | 35 |
| Proiezione Bi-GRU dopo il climate embedding da 8 | 35 | 43 |
| Proiezione meteo e nubi del MAE | 11 | 19 |

Le otto maschere sono concatenate in coda nei profili M1–M3. Il MAE usa le maschere nella proiezione meteorologica e calcola la loss ausiliaria sui valori utilizzabili in questi profili. Entrambi gli imputatori ricevono solo la storia e restituiscono tre indici per timestep.

I checkpoint M0 conservano dimensioni e nomi originali (`history_imputer_best.pt` e `mae_imputer_best.pt`). Il salvataggio standard aggiunge `_m1` per M1 e `_m2` per M2. M3 riusa il checkpoint M2 perché il clipping agisce solo sul futuro. I nuovi checkpoint registrano il profilo storico e la dimensione effettiva. Il caricamento configurato controlla la compatibilità con queste impostazioni.

## Uso e verifica

Dalla radice del repository, in un ambiente Python con le dipendenze installate

~~~bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
~~~

Per costruire il loader secondo la configurazione corrente

~~~python
from src.data.vegetation_dataset import VegetationDataset, collate_vegetation_batch
from src.utils.config import load_config
from torch.utils.data import DataLoader

cfg = load_config()
dataset = VegetationDataset.from_config(cfg, max_files=1, strict_checkpoint=True)
batch = next(iter(DataLoader(dataset, batch_size=4, collate_fn=collate_vegetation_batch)))
~~~

Per verificare il benchmark di ricostruzione sul sottoinsieme interno di validation con checkpoint già disponibili

~~~bash
python -m src.benchmark_imputers --eval-only --max-samples-per-tile 1
~~~

Il benchmark usa lo split di train stratificato per clima con tile disgiunti tra fitting e validation. Salta gli imputatori neurali privi di checkpoint compatibili e usa suffissi per i risultati M1–M3. I tile ID, OOD-S e OOD-T sono riservati alla valutazione finale. Lo scaler distribuito usa l'intero train ufficiale, compresi i tile della validation interna.

### Risultati delle verifiche

Sul train locale sono stati verificati **84 tile, 104.699 finestre e 1.455.984 timestep storici**. I conteggi descrivono occorrenze nelle finestre.

| Variabile | NaN originali | Percentuale storica |
|---|---:|---:|
| `wind` | 0 | 0,00% |
| `humidity` | 106.469 | 7,31% |
| `solar_irradiance` | 64.210 | 4,41% |
| `rainfall` | 60.128 | 4,13% |
| `pressure` | 147.816 | 10,15% |
| `avg_temperature` | 25.457 | 1,75% |
| `min_temperature` | 60.113 | 4,13% |
| `max_temperature` | 60.113 | 4,13% |
| **Totale** | **524.306** | **4,50%** |

Le maschere originali concordano con i NaN storici meteorologici. Mancano interamente sette colonne in `34SFF`, sei in `30UUA` e la pressione in `34SEJ` e `34TFL`. Le inversioni storiche min > media, media > max e min > max sono rispettivamente 529, 237 e 20. Con la soglia di 1 °C il cleaning invalida **852 misure** (284 per ciascuna temperatura, in 284 timestep) e nessun altro valore meteorologico. Per esempio, nella finestra 547 di `33VUF` al 21 giugno 2018 la terna min/media/max è 9/−16/15 °C. In M1 produce valori normalizzati circa −0,2703/−2,3577/−0,9477 con maschere 1/1/1. In M2 la terna elaborata è indisponibile e produce 0/0/0 con maschere 0/0/0.

Sono passati 20 test esistenti e 9 test unitari aggiunti. Uno smoke test su otto finestre reali del train ha verificato tutti i profili del loader e forward, loss e gradienti per Bi-GRU e MAE a 27 e 35 canali. Il benchmark è stato verificato sul sottoinsieme interno del train con pesi temporanei e al più una finestra per tile. Questi controlli misurano la corretta esecuzione. I risultati del confronto M0–M3 e del forecasting restano da misurare.

## Autori

- Nicole Bovolenta
- Daniele Cantagallo
- Gabriele Orazio Di Raimondo
