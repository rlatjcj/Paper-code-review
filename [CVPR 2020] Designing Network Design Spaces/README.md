# Designing Network Design Spaces

## Summary
[notion](https://www.notion.so/Designing-Network-Design-Spaces-455b9494747c46a29b3b6eb9e70425c0)
[pdf](https://github.com/DeepPaperStudy/DPS-5th/blob/master/20200530-Designing%20Network%20Design%20Spaces-SungchulKim.pdf)  

## Code
[official(pytorch)](https://github.com/facebookresearch/pycls)

## Implementation
[code baseline](https://github.com/rlatjcj/code_baseline)
### Search Design Space
```
python search_space.py --config ./config.yml --num-model 500 --baseline-path /path/of/code_baseline --model-name AnyNetXA
```

### Train models 
```
sh ./main.sh
```
`main.sh` is already set to default AnyNet settings.