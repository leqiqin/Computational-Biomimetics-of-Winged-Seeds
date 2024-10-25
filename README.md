# Computational Biomimetics of Winged Seeds
![teaser](imgs/teaser.png "teaser")
![gallery](imgs/gallery.png "gallery")
This codebase contains the research code associated with the paper "[Computational Biomimetics of Winged Seeds
](https://leqiqin.github.io/publication/seeds2024)" (ACM Transactions on Graphics, SIGGRAPH Asia 2024).
## System requirement
We compiled and tested this codebase on the following system:
- Ubuntu 22.04;
- Python 3.10;
- CUDA 11.8
## Installation
```
conda env create -f environment.yaml
```

## Running Optimization Experiments
### Sec 9.1 Descent Experiment
```
python optimize_falling.py
```
### Sec 9.2 Rotational Acceleration Experiment
```
python optimize_rotation.py
```
### Sec 9.3 Expectation Comparison Experiment
```
python optimize_expectation.py
```
### Sec 9.4 Regression Experiment
```
python optimize_regression.py
```
### Sec 9.5 Fabrication Experiment
```
cd fabrication
python optimize_fabrication.py
```

## Visualization of the Shape Space
Shooting data of the seeds are included in the data/shooting directory. We provide a visualization program based on Polyscope, which allows you to explore the shape space spanned by the seeds. To visualize the shape space, run
```
python shape_space.py
```
You can modify the interpolation parameters in the upper right corner. 
![shapespace](imgs/shapespace.png "shapespace")Note that the current visualization program only includes a subset of seeds; to include others, you can modify the shape_space.py file.

## Download Datasets
Download original scanned winged-seeds models [here](datasets/winged_seeds_models_scan.zip).

Download winged-seeds models recontructed by LDDMM shooting [here](datasets/winged_seeds_models_shoot.zip).

Each shooting mesh is denoted by "seed{a}{b}.ply", where {a} is an identifier for its species and {b} represents its sequential number within that species. 

### Species Identifier and its scientific name. 

| Identifier | Species                    |
|------------|----------------------------|
| 0          | Bluebird Vine              |
| 1          | Hopea hainanensis          |
| 2          | Engelhardia spicata        |
| 3          | Spinyleaf pricklyash       |
| 4          | Congea tomentosa           |
| 5          | Hiptage lucida             |
| 6          | Illigera rhodantha         |
| 7          | Securidaca inappendiculata |
| 8          | Tipuana tipu               |
| 9          | Loeseneriella yunnanensis  |
| 10         | Pterocymbium               |
| 11         | Pterygota alata            |
| 12         | Scaphium wallichii         |
| 13         | Ventilago leiocarpa        |

### Visualization of the Shooting Dataset
<style>
.grid-container {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0px;
    padding: 0;
}
.grid-item {
    padding: 0;
    margin: 0;
    line-height: 0;
}
.grid-item img {
    width: 100%;
    display: block;
}
</style>

<div class="grid-container">
<div class="grid-item"><img src="gif/mesh_1.gif" alt="mesh_1.gif" title="mesh_1.gif"></div>
<div class="grid-item"><img src="gif/mesh_2.gif" alt="mesh_2.gif" title="mesh_2.gif"></div>
<div class="grid-item"><img src="gif/mesh_3.gif" alt="mesh_3.gif" title="mesh_3.gif"></div>
<div class="grid-item"><img src="gif/mesh_4.gif" alt="mesh_4.gif" title="mesh_4.gif"></div>
<div class="grid-item"><img src="gif/mesh_5.gif" alt="mesh_5.gif" title="mesh_5.gif"></div>
<div class="grid-item"><img src="gif/mesh_6.gif" alt="mesh_6.gif" title="mesh_6.gif"></div>
<div class="grid-item"><img src="gif/mesh_7.gif" alt="mesh_7.gif" title="mesh_7.gif"></div>
<div class="grid-item"><img src="gif/mesh_8.gif" alt="mesh_8.gif" title="mesh_8.gif"></div>
<div class="grid-item"><img src="gif/mesh_9.gif" alt="mesh_9.gif" title="mesh_9.gif"></div>
<div class="grid-item"><img src="gif/mesh_10.gif" alt="mesh_10.gif" title="mesh_10.gif"></div>
<div class="grid-item"><img src="gif/mesh_11.gif" alt="mesh_11.gif" title="mesh_11.gif"></div>
<div class="grid-item"><img src="gif/mesh_12.gif" alt="mesh_12.gif" title="mesh_12.gif"></div>
<div class="grid-item"><img src="gif/mesh_13.gif" alt="mesh_13.gif" title="mesh_13.gif"></div>
<div class="grid-item"><img src="gif/mesh_14.gif" alt="mesh_14.gif" title="mesh_14.gif"></div>
<div class="grid-item"><img src="gif/mesh_15.gif" alt="mesh_15.gif" title="mesh_15.gif"></div>
<div class="grid-item"><img src="gif/mesh_16.gif" alt="mesh_16.gif" title="mesh_16.gif"></div>
<div class="grid-item"><img src="gif/mesh_17.gif" alt="mesh_17.gif" title="mesh_17.gif"></div>
<div class="grid-item"><img src="gif/mesh_18.gif" alt="mesh_18.gif" title="mesh_18.gif"></div>
<div class="grid-item"><img src="gif/mesh_19.gif" alt="mesh_19.gif" title="mesh_19.gif"></div>
<div class="grid-item"><img src="gif/mesh_20.gif" alt="mesh_20.gif" title="mesh_20.gif"></div>
<div class="grid-item"><img src="gif/mesh_21.gif" alt="mesh_21.gif" title="mesh_21.gif"></div>
<div class="grid-item"><img src="gif/mesh_22.gif" alt="mesh_22.gif" title="mesh_22.gif"></div>
<div class="grid-item"><img src="gif/mesh_23.gif" alt="mesh_23.gif" title="mesh_23.gif"></div>
<div class="grid-item"><img src="gif/mesh_24.gif" alt="mesh_24.gif" title="mesh_24.gif"></div>
<div class="grid-item"><img src="gif/mesh_25.gif" alt="mesh_25.gif" title="mesh_25.gif"></div>
<div class="grid-item"><img src="gif/mesh_26.gif" alt="mesh_26.gif" title="mesh_26.gif"></div>
<div class="grid-item"><img src="gif/mesh_27.gif" alt="mesh_27.gif" title="mesh_27.gif"></div>
<div class="grid-item"><img src="gif/mesh_28.gif" alt="mesh_28.gif" title="mesh_28.gif"></div>
<div class="grid-item"><img src="gif/mesh_29.gif" alt="mesh_29.gif" title="mesh_29.gif"></div>
<div class="grid-item"><img src="gif/mesh_30.gif" alt="mesh_30.gif" title="mesh_30.gif"></div>
<div class="grid-item"><img src="gif/mesh_31.gif" alt="mesh_31.gif" title="mesh_31.gif"></div>
<div class="grid-item"><img src="gif/mesh_32.gif" alt="mesh_32.gif" title="mesh_32.gif"></div>
<div class="grid-item"><img src="gif/mesh_33.gif" alt="mesh_33.gif" title="mesh_33.gif"></div>
<div class="grid-item"><img src="gif/mesh_34.gif" alt="mesh_34.gif" title="mesh_34.gif"></div>
<div class="grid-item"><img src="gif/mesh_35.gif" alt="mesh_35.gif" title="mesh_35.gif"></div>
<div class="grid-item"><img src="gif/mesh_36.gif" alt="mesh_36.gif" title="mesh_36.gif"></div>
<div class="grid-item"><img src="gif/mesh_37.gif" alt="mesh_37.gif" title="mesh_37.gif"></div>
<div class="grid-item"><img src="gif/mesh_38.gif" alt="mesh_38.gif" title="mesh_38.gif"></div>
<div class="grid-item"><img src="gif/mesh_39.gif" alt="mesh_39.gif" title="mesh_39.gif"></div>
<div class="grid-item"><img src="gif/mesh_40.gif" alt="mesh_40.gif" title="mesh_40.gif"></div>
<div class="grid-item"><img src="gif/mesh_41.gif" alt="mesh_41.gif" title="mesh_41.gif"></div>
<div class="grid-item"><img src="gif/mesh_42.gif" alt="mesh_42.gif" title="mesh_42.gif"></div>
<div class="grid-item"><img src="gif/mesh_43.gif" alt="mesh_43.gif" title="mesh_43.gif"></div>
<div class="grid-item"><img src="gif/mesh_44.gif" alt="mesh_44.gif" title="mesh_44.gif"></div>
<div class="grid-item"><img src="gif/mesh_45.gif" alt="mesh_45.gif" title="mesh_45.gif"></div>
<div class="grid-item"><img src="gif/mesh_46.gif" alt="mesh_46.gif" title="mesh_46.gif"></div>
<div class="grid-item"><img src="gif/mesh_47.gif" alt="mesh_47.gif" title="mesh_47.gif"></div>
<div class="grid-item"><img src="gif/mesh_48.gif" alt="mesh_48.gif" title="mesh_48.gif"></div>
<div class="grid-item"><img src="gif/mesh_49.gif" alt="mesh_49.gif" title="mesh_49.gif"></div>
<div class="grid-item"><img src="gif/mesh_50.gif" alt="mesh_50.gif" title="mesh_50.gif"></div>
<div class="grid-item"><img src="gif/mesh_51.gif" alt="mesh_51.gif" title="mesh_51.gif"></div>
<div class="grid-item"><img src="gif/mesh_52.gif" alt="mesh_52.gif" title="mesh_52.gif"></div>
<div class="grid-item"><img src="gif/mesh_53.gif" alt="mesh_53.gif" title="mesh_53.gif"></div>
<div class="grid-item"><img src="gif/mesh_54.gif" alt="mesh_54.gif" title="mesh_54.gif"></div>
<div class="grid-item"><img src="gif/mesh_55.gif" alt="mesh_55.gif" title="mesh_55.gif"></div>
</div>