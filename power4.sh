#! bin/sh

for size in 16; do
    for thr in 0.8; do
        for signal in 4.0; do
            for alpha in 0.05; do
                for workers in 16; do
                    for iter in 110; do
                        for seed in {0..9}; do
                            qsub power4.csh ${size} ${thr} ${signal} ${alpha} ${workers} ${iter} ${seed}
                        done
                    done
                done
            done
        done
    done
done