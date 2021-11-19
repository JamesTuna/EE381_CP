dropout=0
for head in "8" "16"
do
    for n_Elayer in "2" "4"
    do
        for n_Tlayer in "4" "8"
        do
            for Etype in "GRU" "LSTM"
            do
                trial="${n_Tlayer}Tlayer[h${head}]_${n_Elayer}${Etype}_${dropout}dropout"
                echo $trial
                python3 runModel.py --n_Tlayer ${n_Tlayer} --n_head ${head} --n_Elayer ${n_Elayer} \
                                    --Etype ${Etype} --dropout ${dropout} &> "${trial}.log"
                
                trial="${n_Tlayer}Tlayer[h${head}]_${n_Elayer}${Etype}_conv_${dropout}dropout"
                echo $trial
                python3 runModel.py --n_Tlayer ${n_Tlayer} --n_head ${head} --n_Elayer ${n_Elayer} \
                                    --Etype ${Etype} --dropout ${dropout} --use_conv &> "${trial}.log"
            done
        done
    done
done
