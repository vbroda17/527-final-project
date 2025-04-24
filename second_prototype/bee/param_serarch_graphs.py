import pandas as pd
import matplotlib.pyplot as plt
import os

# Load results and failures
results = pd.read_csv('bco_experiments/results.csv')
failures = pd.read_csv('bco_experiments/failures.csv')

# Aggregate metrics per parameter combination
agg = results.groupby(['bees','scout_frac','extra_food']).agg(
    max_fit=('best_fit','max'),
    avg_fit=('best_fit','mean'),
    avg_path=('path_length_AU','mean'),
    avg_steps=('arrival_step','mean'),
    n_runs=('run_idx','size')
).reset_index()

# Prepare output directory
out_dir = 'bco_experiments/analysis_plots'
os.makedirs(out_dir, exist_ok=True)

# Plot average fitness vs. number of bees for each combination of scout_frac and extra_food
for ef in sorted(agg.extra_food.unique()):
    sub = agg[agg.extra_food == ef]
    plt.figure()
    for sf in sorted(sub.scout_frac.unique()):
        sf_sub = sub[sub.scout_frac == sf]
        plt.plot(sf_sub.bees, sf_sub.avg_fit, label=f'scout_frac={sf}')
    plt.title(f'Average Fitness vs Bees (extra_food={ef})')
    plt.xlabel('Number of Bees')
    plt.ylabel('Average Best Fitness')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'avg_fitness_extra{ef}.png'))
    plt.close()

# Plot max fitness vs. number of bees
for ef in sorted(agg.extra_food.unique()):
    sub = agg[agg.extra_food == ef]
    plt.figure()
    for sf in sorted(sub.scout_frac.unique()):
        sf_sub = sub[sub.scout_frac == sf]
        plt.plot(sf_sub.bees, sf_sub.max_fit, label=f'scout_frac={sf}')
    plt.title(f'Max Fitness vs Bees (extra_food={ef})')
    plt.xlabel('Number of Bees')
    plt.ylabel('Max Best Fitness')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'max_fitness_extra{ef}.png'))
    plt.close()

# Plot average path length vs. number of bees
for ef in sorted(agg.extra_food.unique()):
    sub = agg[agg.extra_food == ef]
    plt.figure()
    for sf in sorted(sub.scout_frac.unique()):
        sf_sub = sub[sub.scout_frac == sf]
        plt.plot(sf_sub.bees, sf_sub.avg_path, label=f'scout_frac={sf}')
    plt.title(f'Average Path Length vs Bees (extra_food={ef})')
    plt.xlabel('Number of Bees')
    plt.ylabel('Average Path Length (AU)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'avg_path_extra{ef}.png'))
    plt.close()

# Plot average arrival steps vs. number of bees
for ef in sorted(agg.extra_food.unique()):
    sub = agg[agg.extra_food == ef]
    plt.figure()
    for sf in sorted(sub.scout_frac.unique()):
        sf_sub = sub[sub.scout_frac == sf]
        plt.plot(sf_sub.bees, sf_sub.avg_steps, label=f'scout_frac={sf}')
    plt.title(f'Average Arrival Step vs Bees (extra_food={ef})')
    plt.xlabel('Number of Bees')
    plt.ylabel('Average Arrival Step')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'avg_steps_extra{ef}.png'))
    plt.close()

# Plot failure rate vs. number of bees
failure_rates = failures.groupby(['bees','scout_frac','extra_food']).size().reset_index(name='failures')
failure_rates = failure_rates.merge(
    agg[['bees','scout_frac','extra_food','n_runs']],
    on=['bees','scout_frac','extra_food']
)
failure_rates['failure_rate'] = failure_rates.failures / failure_rates.n_runs

for ef in sorted(failure_rates.extra_food.unique()):
    sub = failure_rates[failure_rates.extra_food == ef]
    plt.figure()
    for sf in sorted(sub.scout_frac.unique()):
        sf_sub = sub[sub.scout_frac == sf]
        plt.plot(sf_sub.bees, sf_sub.failure_rate, label=f'scout_frac={sf}')
    plt.title(f'Failure Rate vs Bees (extra_food={ef})')
    plt.xlabel('Number of Bees')
    plt.ylabel('Failure Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'failure_rate_extra{ef}.png'))
    plt.close()

print(f"Analysis plots saved in {out_dir}")