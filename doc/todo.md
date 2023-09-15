- build cached version
  - separate softmax and value aggregation
  - add offset for rope in case of cache
  
v transpose vocab embedding for final mult
v move to dim first in shape
v matrix col first as a result (typically L over row dimension, dim over col due to mat mul)
v check shape sizes, fix shapes in reading
