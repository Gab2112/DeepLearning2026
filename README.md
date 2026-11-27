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
| M3 | Come M1 | Attivo | Clipping delle variabili meteo con unità verificate |
| M4 | Come M0, con otto maschere di disponibilità originale | Disattivo | Come M1 |

`src/data/weather_preprocessor.py` lavora su copie delle 18 covariate. M0 e M4 condividono nel loader l'interpolazione lineare prima di Arcsinh e il fallback a zero grezzo per le colonne interamente mancanti. Il riempimento degli eventuali NaN residui avviene in scala normalizzata. Negli imputatori neurali M0 mantiene il percorso storico di normalizzazione e zero filling. M4 applica l'interpolazione anche a questi imputatori.

Una misura è disponibile quando è finita, osservata nella maschera sorgente e supera gli eventuali controlli di qualità. I flag valgono 1 per una misura originale utilizzabile e 0 per una misura indisponibile. In M1–M3 i mancanti storici ricevono zero nello spazio normalizzato. In M4 il flag rimane 0 anche se l'interpolatore ha prodotto un valore.

Il dataset espone quattro maschere storiche a otto colonne. `history_weather_original_missing_mask` unisce i NaN e la maschera sorgente. `history_weather_invalid_mask` registra le anomalie aggiunte dal cleaning. `history_weather_interpolated_mask` indica i valori ricavati da osservazioni della stessa colonna. Le colonne interamente mancanti mantengono questa maschera a 0. `history_weather_mask` indica la disponibilità effettiva del valore originale. Per il futuro il loader espone `future_weather_original_missing_mask` e `future_weather_mask`. Le maschere meteorologiche, la maschera delle osservazioni vegetative e le padding mask sono tensori distinti.

### Cleaning storico e clipping futuro

La scheda E-OBS indica vento in m/s, umidità relativa in percentuale, irraggiamento in W/m², precipitazioni giornaliere in mm, pressione a livello del mare in hPa e temperature in °C. Il cleaning storico segnala valori non finiti ulteriori rispetto ai NaN originali, vento, pioggia o irraggiamento negativi, umidità fuori da [0, 100] e pressione non positiva. Gli estremi statistici finiti entro queste regole restano disponibili. [Unità delle variabili E-OBS](https://cds.climate.copernicus.eu/datasets/insitu-gridded-observations-europe?tab=overview)

Per le temperature il controllo confronta i tre possibili ordini tra coppie disponibili. `temperature_tolerance_c: 1.0` è una soglia sperimentale. Con `temperature_policy: "mask_triplet"` un'inversione che supera la tolleranza invalida tutti i membri ancora disponibili della terna. `temperature_policy: "ignore"` disattiva il solo controllo termico. I valori invalidati passano a NaN nella copia elaborata prima della normalizzazione.

In M0, M1, M2 e M4 il futuro usa direttamente i valori perturbati disponibili. M3 porta a zero vento, pioggia e irraggiamento negativi e limita l'umidità a [0, 100]. La documentazione locale descrive le colonne nuvolose come frazioni. La fonte GreenEarthNet documenta la maschera per pixel e lascia indeterminata la scala delle tre colonne aggregate di questi NPZ. Per questo il clipping delle nubi è sospeso. M3 conserva anche i valori perturbati di pressione e temperatura e mantiene la disponibilità dei valori sottoposti a clipping. Il loader usa `future` e `future_mask` ed esclude `future_noise` e le colonne vegetative future. I target indisponibili restano NaN e sono segnalati da `target_mask`. [Documentazione GreenEarthNet](https://github.com/vitusbenson/greenearthnet) · [Descrizione delle colonne locali](DATA/README.md#feature)

## Scaler e split riproducibile

La slide 5 della traccia consente di usare uno scaler proprio. Con `scaling.mode: "fitting_tiles"` il codice usa lo split interno geografico e stratificato per clima, con seed 42, 70 tile di fitting e 14 di validation. Calcola media e varianza di popolazione sulle 21 colonne storiche utilizzabili e sui tre target osservati dei tile di fitting. Maschere sorgente, valori non finiti e invalidazioni M2/M3 escludono le rispettive misure. Le statistiche precedono l'interpolazione e il riempimento.

Lo stesso scaler trasforma fitting, validation e covariate future. M0, M1 e M4 condividono le statistiche di fitting. M2 e M3 condividono le statistiche con cleaning storico. Il manifest e lo scaler salvano seed, tile e impostazioni di cleaning. I checkpoint incorporano anche le statistiche effettive. La modalità esplicita `distributed` mantiene la compatibilità con lo scaler e i checkpoint distribuiti in precedenza. Il file originale è conservato. Gli split ID, OOD-S e OOD-T sono riservati alla valutazione finale. [Traccia, slide 4 e 5](DL_Project.pdf)

## Dimensioni di input e checkpoint

| Componente | M0 | M1–M4 |
|---|---:|---:|
| `VegetationDataset` storico | 21 | 29 |
| `VegetationDataset` futuro | 18 | 26 |
| Input storico degli imputatori neurali | 27 | 35 |
| Proiezione Bi-GRU dopo il climate embedding da 8 | 35 | 43 |
| Proiezione meteo e nubi del MAE | 11 | 19 |

Le otto maschere sono concatenate in coda nei profili M1–M4. Il MAE usa le maschere nella proiezione meteorologica e calcola la loss ausiliaria sui valori utilizzabili in questi profili. Entrambi gli imputatori ricevono solo la storia e restituiscono tre indici per timestep.

I checkpoint con scaler di fitting usano il suffisso `_fit`, preceduto dal suffisso del profilo storico quando previsto. Per esempio, M2 usa `history_imputer_best_m2_fit.pt` e `mae_imputer_best_m2_fit.pt`. M3 riusa il checkpoint M2 perché il clipping agisce solo sul futuro. In inferenza i wrapper recuperano lo scaler incorporato. I vecchi checkpoint M0 a 27 canali mantengono i nomi originali in modalità `distributed`.

## Uso e verifica

Dalla radice del repository, in un ambiente Python con le dipendenze installate

~~~bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
~~~

Per preparare lo studio senza addestrare modelli

~~~bash
python -m src.experiment_protocol prepare --output-dir runs/weather-study --imputer-profile M2 --candidate M2:mae --candidate M4:linear
python -m src.report_weather_qc --output runs/weather-study/weather_qc_train.csv
~~~

### A. Profili meteo con imputatore fisso

I file `A_m*_linear.yaml` usano tutti l'interpolatore lineare. M4 e M0 isolano l'aggiunta delle maschere. M1 e M4 confrontano l'effetto dell'interpolazione a parità di maschere. M2 e M1 confrontano il cleaning, comprese le rispettive statistiche. M3 e M2 isolano il clipping futuro. Questi sono confronti da misurare con la stessa architettura, i medesimi tile e budget quando il forecaster sarà disponibile.

~~~python
from src.data.splits import get_experiment_split
from src.data.vegetation_dataset import VegetationDataset
from src.utils.config import load_config

cfg = load_config("runs/weather-study/A_m4_linear.yaml")
fit_tiles, validation_tiles, _ = get_experiment_split(cfg)
train = VegetationDataset.from_config(cfg, file_list=fit_tiles, strict_checkpoint=True)
validation = VegetationDataset.from_config(cfg, file_list=validation_tiles, strict_checkpoint=True)
~~~

### B. Imputatori con preprocessing fisso

Con `B_m2_imputers.yaml` tutti i candidati usano lo stesso profilo meteo, lo stesso split e lo stesso seed. L'opzione `all_obs` nasconde a turno ogni osservazione clear-sky disponibile. Ogni imputatore vede le stesse finestre e riceve NaN e maschera sulle tre etichette nascoste. Il report registra MAE, RMSE, WMAPE, numero di date nascoste, tempo di training comprensivo della validation e tempo di inferenza. Il benchmark definitivo richiede entrambi i checkpoint neurali con scaler compatibile, stato dell'ottimizzatore e metadati di training completato.

Questi comandi sono per la fase definitiva

~~~bash
python -m src.benchmark_imputers --config runs/weather-study/B_m2_imputers.yaml --train-all --eval-mode all_obs
python -m src.benchmark_imputers --config runs/weather-study/B_m2_imputers.yaml --eval-only --eval-mode all_obs
~~~

Il primo comando addestra e valuta. Il secondo ripete la sola valutazione quando i checkpoint sono disponibili.

Per verificare soltanto l'esecuzione usare `--eval-only --smoke --max-samples-per-tile 1`. I checkpoint assenti vengono segnalati e gli artefatti hanno il suffisso `_smoke`.

### C. Combinazioni da valutare sul forecasting

L'opzione `--candidate PROFILO:STRATEGIA` crea le configurazioni `C_*.yaml` per i candidati indicati. Quando il forecaster sarà disponibile, esportare un NPZ con `y_pred_quantiles` di forma `(finestre, 3 step, 3 indici, 3 quantili)`, `tile_names` senza estensione e `window_indices` a partire da zero. L'ordine è quello dei tile di validation del manifest, seguito dall'ordine delle finestre nei file. Il valutatore verifica gli identificativi e calcola pinball loss, MAE della mediana, coverage e calibrazione empirica `P(y <= q_pred)` per quantile e indice. Usare `--normalized` se le predizioni sono in scala Arcsinh.

~~~bash
python -m src.experiment_protocol score-forecast --config runs/weather-study/C_m2_mae.yaml --predictions runs/weather-study/predictions_m2_mae.npz --output runs/weather-study/forecast_m2_mae.json
~~~

## Risultati verificati e limitazioni

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

Le maschere originali concordano con i NaN storici meteorologici. Mancano interamente sette colonne in `34SFF`, sei in `30UUA` e la pressione in `34SEJ` e `34TFL`. Con la soglia di 1 °C il cleaning invalida **852 misure precedentemente utilizzabili** in 284 timestep di 15 tile. Sono 284 misure per ciascuna delle tre temperature e 819 misure nel fitting. Nella finestra 547 di `33VUF` al 21 giugno 2018 la terna min/media/max di 9/−16/15 °C produce circa −0,3424/−2,3862/−1,0030 in M1 con maschere 1/1/1. In M2 la terna è invalidata e produce 0/0/0 con maschere 0/0/0. [Riepilogo per tile e variabile](results/weather_qc_train.csv)

Le righe MAE, lineare e spline del benchmark archiviato coincidono. L'uguaglianza tra fallback MAE e lineare è stata riprodotta su 24 date nascoste in otto finestre del train senza checkpoint MAE. Anche lo spline usa l'interpolazione lineare con due osservazioni disponibili. Restano da verificare i checkpoint e i log della corsa archiviata per attribuire con certezza il risultato MAE al fallback.

La suite completa ha superato 35 test. Uno smoke test su otto finestre reali ha verificato i profili M0–M4 con scaler di fitting. Un benchmark smoke sulla validation interna ha valutato otto metodi classici su 14 date e segnalato i due checkpoint neurali assenti. Una prova sintetica con un passo di ottimizzazione per modello ha verificato salvataggio, round-trip dello scaler e tempi. La fase sperimentale con checkpoint addestrati e forecaster resta da eseguire.

## Autori

- Nicole Bovolenta
- Daniele Cantagallo
- Gabriele Orazio Di Raimondo
