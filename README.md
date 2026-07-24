# ASTRA para Quijotes

Este repositorio conserva el pipeline ASTRA de DESI y añade una entrada para
catálogos de halos FoF de las simulaciones Quijotes.

Para cada realización, el flujo:

1. lee `GroupPos`, `GroupMass`, `GroupVel` y `GroupLen` con
   `readfof.FoF_catalog`;
2. convierte las unidades igual que `read_data.ipynb`;
3. genera un catálogo uniforme independiente, del mismo tamaño que el real,
   para cada iteración;
4. calcula el grafo de Delaunay, `NDATA`, `NRAND`, la clase ASTRA y las
   probabilidades finales;
5. escribe los mismos productos raw, pairs, classification y probabilities.

## Verificación rápida

Desde la raíz del repositorio:

```bash
python astra/main.py \
  --release QUIJOTES \
  --cat-dir "$HOME/Desktop/Quijotes/data/fiducial" \
  --snapnum 3 \
  --simulation-id 0 \
  --n-iterations 2 \
  --max-halos 256 \
  --n-jobs 1 \
  --raw-out outputs/check/raw \
  --class-out outputs/check/astra \
  --groups-out outputs/check/groups
```

`--max-halos` existe únicamente para pruebas cortas. Una corrida científica
usa el catálogo completo y las 100 iteraciones:

```bash
python astra/main.py \
  --release QUIJOTES \
  --cat-dir "$HOME/Desktop/Quijotes/data/fiducial" \
  --snapnum 3 \
  --simulation-id 0 \
  --n-iterations 100 \
  --raw-out outputs/fiducial_000/raw \
  --class-out outputs/fiducial_000/astra \
  --groups-out outputs/fiducial_000/groups
```

`--n-random` sigue disponible como alias histórico de `--n-iterations`. En
cada iteración se generan exactamente `N_halos` puntos random; no significa
el número de puntos de una sola realización.

## Parámetros de Quijotes

- `--cat-dir`: directorio que contiene `groups_NNN`.
- `--snapnum`: snapshot a leer; el default es `3`.
- `--simulation-id`: realización a procesar. El lector reconoce tanto el
  layout estándar de Pylians como archivos aplanados
  `group_tab_NNN_<simulation-id>.<part>`.
- `--pylians-library`: directorio que contiene `readfof.py`; por defecto
  `~/Pylians3/library`.
- `--seed`: semilla del único generador NumPy usado para todas las
  realizaciones random; por defecto `42`.
- `--box-min` y `--box-max`: un escalar o tres valores. El default es la caja
  física `[0, 1000)` Mpc/h.
- `--periodic` / `--no-periodic`: usa una caja periódica o abierta.
  La periodicidad está habilitada por defecto.
- `--redshift`: opcional para snapshots no estándar. Para Quijotes se infiere
  el mapa `0→3`, `1→2`, `2→1`, `3→0.5`, `4→0`.

Los nombres reciben automáticamente un tag como `sim000_snap003`, evitando
mezclar realizaciones o snapshots.

## Geometría periódica

El Delaunay ordinario corta vecinos a través de las caras de la caja. La
implementación periódica usa imágenes fantasma cercanas a los bordes y sólo
acepta el resultado cuando un criterio geométrico certifica que ninguna imagen
omitida puede cambiar una celda de Voronoi. Si no logra certificarlo, usa las
27 imágenes completas como fallback exacto.

El modo periódico usa un solo worker por defecto para limitar memoria.
`--n-jobs` permite cambiarlo explícitamente.

`--no-periodic` reproduce la caja abierta usada en la demostración del
notebook, pero introduce efectos de borde y no representa las condiciones
periódicas de Quijotes.

## Unidades y productos

El raw conserva:

- posición: `GroupPos / 1e3`, en Mpc/h;
- masa: `GroupMass * 1e10`, en masas solares/h;
- velocidad física: `GroupVel * (1 + z)`, en km/s;
- número de partículas: `GroupLen`.

Los randoms tienen masa y velocidad `NaN`, `NPART=-1`, IDs `int64` únicos y
`RANDITER=0...N-1`; los halos reales tienen `RANDITER=-1`.

Con la configuración de salida heredada del pipeline, se escriben:

- un raw combinado;
- un archivo pairs combinado;
- un archivo classification por iteración;
- un archivo probability agregado para los halos reales (`iterdata`).

La agrupación FoF posterior y los wedge plots celestes se omiten para Quijotes:
el FoF existente no implementa mínima imagen y los plots requieren RA/DEC/Z.

Los umbrales EDR (`-0.25`, `0.25`, `0.65`) se conservan como defaults y pueden
modificarse con `--r-lower`, `--r-med` y `--r-upper`. Fueron calibrados para
datos observacionales DESI/GAMA, por lo que aplicarlos a Quijotes reproduce el
algoritmo, pero no constituye una recalibración física para las simulaciones.
