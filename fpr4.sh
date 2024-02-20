#! bin/sh

for size in 64; do
    for thr in 0.8; do
        for signal in 0.0; do
            for alpha in 0.05; do
                for workers in 16; do
                    for iter in 120; do
                        for seed in {0..9}; do
                            qsub fpr4.csh ${size} ${thr} ${signal} ${alpha} ${workers} ${iter} ${seed}
                        done
                    done
                done
            done
        done
    done
done