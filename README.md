<h1 align='center'>
  Continual Slow-and-Fast Adaptation of <br>
  Latent Neural Dynamics (CoSFan):<br> 
  Meta-Learning What-How & When to Adapt<br>
<!--   (ICLR 2023 Top-25%)<br> -->
  [<a href='https://openreview.net/forum?id=Dl3MsjaIdp'>OpenReview</a>]
</h1>

<p align='center'>Ryan Missel, Linwei Wang</p>

<p align='center'><img src="https://github.com/user-attachments/assets/20d4f32f-2802-4445-8105-394c850d2527" alt="framework schematic")/></p>
<p align='center'>Figure 1: Overview of <i>CoSFan</i>, showing the <i>what-how</i> meta-model continually aggregate a heterogeneous data stream with a reservoir that identifies tasks via Gaussian mixture models..</p>

## Description
An increasing interest in learning to forecast for time-series of high-dimensional observations is the ability to adapt to systems with diverse underlying dynamics. 
Access to observations that define a stationary distribution of these systems is often unattainable, as the underlying dynamics may change over time. 
Naively training or retraining models at each shift may lead to catastrophic forgetting about previously-seen systems. 
We present a new continual meta-learning (CML) framework to realize continual slow-and fast adaptation of latent dynamics (CoSFan). % by meta-learning <i>what-how \& when</i> to adapt. 
We leverage a feed-forward meta-model to infer <i>what</i> the current system is and <i>how</i> to adapt a latent dynamics function to it, enabling <i>fast adaptation</i> to specific dynamics.
We then develop novel strategies to automatically detect <i>when</i> a shift of data distribution occurs, with which to identify its underlying dynamics and its relation with previously-seen dynamics. 
In combination with fixed-memory experience replay mechanisms, this enables continual <i>slow update</i> of the <i>what-how</i> meta-model.
Empirical studies demonstrated that both the meta- and continual-learning component was critical for learning to forecast across non-stationary distributions of diverse dynamics systems, 
and the feed-forward meta-model combined with task-aware/-relational continual learning strategies significantly outperformed existing CML alternatives.

## Citation
Please cite the following if you use the data or the model in your work:
```bibtex
@inproceedings{
  missel2025cosfan,
  title={Continual Slow-and-Fast Adaptation of Latent Neural Dynamics (Co{SF}an): Meta-Learning What-How \& When to Adapt},
  author={Ryan Missel and Linwei Wang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=Dl3MsjaIdp}
}
```

## Requirements
Refer to <code>requirements.txt</code>.
