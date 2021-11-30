# no feature engineering
# dropout=0
# for head in "8" "16"
# do
#     for n_Elayer in "2" "4"
#     do
#         for n_Tlayer in "4" "8"
#         do
#             for Etype in "GRU" "LSTM"
#             do
#                 trial="${n_Tlayer}Tlayer[h${head}]_${n_Elayer}${Etype}_${dropout}dropout"
#                 echo $trial
#                 python3 runModel.py --n_Tlayer ${n_Tlayer} --n_head ${head} --n_Elayer ${n_Elayer} \
#                                     --Etype ${Etype} --dropout ${dropout} &> "${trial}.log"
                
#                 trial="${n_Tlayer}Tlayer[h${head}]_${n_Elayer}${Etype}_conv_${dropout}dropout"
#                 echo $trial
#                 python3 runModel.py --n_Tlayer ${n_Tlayer} --n_head ${head} --n_Elayer ${n_Elayer} \
#                                     --Etype ${Etype} --dropout ${dropout} --use_conv &> "${trial}.log"
#             done
#         done
#     done
# done

# feature engineering
# T16 h16 GRU4 dropout0 no conv is the best" 0.18 on validation
# for n_Elayer in "4"
# do
#     for head in "16"
#     do
#         for n_Tlayer in "8" "16"
#         do
#             for Etype in "GRU"
#             do
#                 dropout=0
#                 trial="${n_Tlayer}Tlayer[h${head}]_${n_Elayer}${Etype}_${dropout}dropout_largefs"
#                 echo $trial
#                 python3 runModel.py --n_Tlayer ${n_Tlayer} --n_head ${head} --n_Elayer ${n_Elayer} \
#                                     --Etype ${Etype} --dropout ${dropout} --largefs &> "${trial}.log"
                
#                 trial="${n_Tlayer}Tlayer[h${head}]_${n_Elayer}${Etype}_conv_${dropout}dropout_largefs"
#                 echo $trial
#                 python3 runModel.py --n_Tlayer ${n_Tlayer} --n_head ${head} --n_Elayer ${n_Elayer} \
#                                     --Etype ${Etype} --dropout ${dropout} --use_conv --largefs &> "${trial}.log"
                
#                 dropout=0.5
#                 trial="${n_Tlayer}Tlayer[h${head}]_${n_Elayer}${Etype}_${dropout}dropout_largefs"
#                 echo $trial
#                 python3 runModel.py --n_Tlayer ${n_Tlayer} --n_head ${head} --n_Elayer ${n_Elayer} \
#                                     --Etype ${Etype} --dropout ${dropout} --largefs &> "${trial}.log"
                
#                 trial="${n_Tlayer}Tlayer[h${head}]_${n_Elayer}${Etype}_conv_${dropout}dropout_largefs"
#                 echo $trial
#                 python3 runModel.py --n_Tlayer ${n_Tlayer} --n_head ${head} --n_Elayer ${n_Elayer} \
#                                     --Etype ${Etype} --dropout ${dropout} --use_conv --largefs &> "${trial}.log"
#             done
#         done
#     done
# done

# feature engineering
# bagging of 10 models 
n_Elayer=4
n_Tlayer=16
Etype="GRU"
dropout=0
head=16
for seed in "8" "9"
do
    trial="${n_Tlayer}Tlayer[h${head}]_${n_Elayer}${Etype}_${dropout}dropout_largefs_embd256_boot${seed}"
    echo $trial
    python3 runModel256.py --n_Tlayer ${n_Tlayer} --n_head ${head} --n_Elayer ${n_Elayer} \
                                     --Etype ${Etype} --dropout ${dropout} --largefs --bootstrap \
                                     --bootstrap_seed $seed &> "${trial}.log"
done
# no bootstrap
trial="${n_Tlayer}Tlayer[h${head}]_${n_Elayer}${Etype}_${dropout}dropout_largefs_embd256"
echo $trial
python3 runModel256.py --n_Tlayer ${n_Tlayer} --n_head ${head} --n_Elayer ${n_Elayer} \
                                 --Etype ${Etype} --dropout ${dropout} --largefs &> "${trial}.log"

