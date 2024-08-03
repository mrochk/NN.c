#ifndef LOSS_H
#define LOSS_H

#include "../tensors/tensors.h"

float MSE(Vector preds, Vector targets); 

float MAE(Vector preds, Vector targets); 

#endif