       ЃK"	  Xі(зAbrain.Event:2@Rиu_     ЎE	&ћЕXі(зA"шО

input_inputPlaceholder*
dtype0*1
_output_shapes
:џџџџџџџџџИа*&
shape:џџџџџџџџџИа
s
input/random_uniform/shapeConst*
_output_shapes
:*%
valueB"             *
dtype0
]
input/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *OSО
]
input/random_uniform/maxConst*
valueB
 *OS>*
dtype0*
_output_shapes
: 
Ќ
"input/random_uniform/RandomUniformRandomUniforminput/random_uniform/shape*
T0*
dtype0*&
_output_shapes
: *
seed2пу*
seedБџх)
t
input/random_uniform/subSubinput/random_uniform/maxinput/random_uniform/min*
T0*
_output_shapes
: 

input/random_uniform/mulMul"input/random_uniform/RandomUniforminput/random_uniform/sub*&
_output_shapes
: *
T0

input/random_uniformAddinput/random_uniform/mulinput/random_uniform/min*
T0*&
_output_shapes
: 

input/kernel
VariableV2*
dtype0*&
_output_shapes
: *
	container *
shape: *
shared_name 
М
input/kernel/AssignAssigninput/kernelinput/random_uniform*
use_locking(*
T0*
_class
loc:@input/kernel*
validate_shape(*&
_output_shapes
: 
}
input/kernel/readIdentityinput/kernel*
T0*
_class
loc:@input/kernel*&
_output_shapes
: 
X
input/ConstConst*
_output_shapes
: *
valueB *    *
dtype0
v

input/bias
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
Ё
input/bias/AssignAssign
input/biasinput/Const*
use_locking(*
T0*
_class
loc:@input/bias*
validate_shape(*
_output_shapes
: 
k
input/bias/readIdentity
input/bias*
T0*
_class
loc:@input/bias*
_output_shapes
: 
p
input/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
х
input/convolutionConv2Dinput_inputinput/kernel/read*1
_output_shapes
:џџџџџџџџџИа *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME

input/BiasAddBiasAddinput/convolutioninput/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа 

leaky_re_lu_1/LeakyRelu	LeakyReluinput/BiasAdd*
alpha%>*1
_output_shapes
:џџџџџџџџџИа *
T0
f
$dropout_1/keras_learning_phase/inputConst*
value	B
 Z *
dtype0
*
_output_shapes
: 

dropout_1/keras_learning_phasePlaceholderWithDefault$dropout_1/keras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 

dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
T0
*
_output_shapes
: 
c
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
: 
s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?

dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*1
_output_shapes
:џџџџџџџџџИа *
T0
й
dropout_1/cond/mul/SwitchSwitchleaky_re_lu_1/LeakyReludropout_1/cond/pred_id*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа *
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu
z
dropout_1/cond/dropout/rateConst^dropout_1/cond/switch_t*
valueB
 *  >*
dtype0*
_output_shapes
: 
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:
{
dropout_1/cond/dropout/sub/xConst^dropout_1/cond/switch_t*
_output_shapes
: *
valueB
 *  ?*
dtype0
}
dropout_1/cond/dropout/subSubdropout_1/cond/dropout/sub/xdropout_1/cond/dropout/rate*
_output_shapes
: *
T0

)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ъ
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
T0*
dtype0*1
_output_shapes
:џџџџџџџџџИа *
seed2н*
seedБџх)
Ї
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ь
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*1
_output_shapes
:џџџџџџџџџИа *
T0
О
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*1
_output_shapes
:џџџџџџџџџИа 
 
dropout_1/cond/dropout/addAdddropout_1/cond/dropout/sub%dropout_1/cond/dropout/random_uniform*
T0*1
_output_shapes
:џџџџџџџџџИа 
}
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*1
_output_shapes
:џџџџџџџџџИа *
T0

dropout_1/cond/dropout/truedivRealDivdropout_1/cond/muldropout_1/cond/dropout/sub*
T0*1
_output_shapes
:џџџџџџџџџИа 

dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/truedivdropout_1/cond/dropout/Floor*
T0*1
_output_shapes
:џџџџџџџџџИа 
з
dropout_1/cond/Switch_1Switchleaky_re_lu_1/LeakyReludropout_1/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа 

dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
T0*
N*3
_output_shapes!
:џџџџџџџџџИа : 
Ц
max_pooling2d_1/MaxPoolMaxPooldropout_1/cond/Merge*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ 
v
conv2d_1/random_uniform/shapeConst*
_output_shapes
:*%
valueB"          @   *
dtype0
`
conv2d_1/random_uniform/minConst*
valueB
 *ЋЊЊН*
dtype0*
_output_shapes
: 
`
conv2d_1/random_uniform/maxConst*
valueB
 *ЋЊЊ=*
dtype0*
_output_shapes
: 
В
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
dtype0*&
_output_shapes
: @*
seed2їЄѓ*
seedБџх)*
T0
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
T0*
_output_shapes
: 

conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*&
_output_shapes
: @*
T0

conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*&
_output_shapes
: @*
T0

conv2d_1/kernel
VariableV2*
dtype0*&
_output_shapes
: @*
	container *
shape: @*
shared_name 
Ш
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
: @

conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @
[
conv2d_1/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_1/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
­
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
s
"conv2d_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ї
conv2d_1/convolutionConv2Dmax_pooling2d_1/MaxPoolconv2d_1/kernel/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ@*
	dilations
*
T0

conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџЈ@

leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_1/BiasAdd*
alpha%>*1
_output_shapes
:џџџџџџџџџЈ@*
T0

dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
T0
*
_output_shapes
: 
c
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
: *
T0

s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*
T0*1
_output_shapes
:џџџџџџџџџЈ@
й
dropout_2/cond/mul/SwitchSwitchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu
z
dropout_2/cond/dropout/rateConst^dropout_2/cond/switch_t*
valueB
 *  >*
dtype0*
_output_shapes
: 
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
T0*
out_type0*
_output_shapes
:
{
dropout_2/cond/dropout/sub/xConst^dropout_2/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
}
dropout_2/cond/dropout/subSubdropout_2/cond/dropout/sub/xdropout_2/cond/dropout/rate*
T0*
_output_shapes
: 

)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ъ
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
T0*
dtype0*1
_output_shapes
:џџџџџџџџџЈ@*
seed2из*
seedБџх)
Ї
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ь
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*1
_output_shapes
:џџџџџџџџџЈ@*
T0
О
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*
T0*1
_output_shapes
:џџџџџџџџџЈ@
 
dropout_2/cond/dropout/addAdddropout_2/cond/dropout/sub%dropout_2/cond/dropout/random_uniform*
T0*1
_output_shapes
:џџџџџџџџџЈ@
}
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*
T0*1
_output_shapes
:џџџџџџџџџЈ@

dropout_2/cond/dropout/truedivRealDivdropout_2/cond/muldropout_2/cond/dropout/sub*
T0*1
_output_shapes
:џџџџџџџџџЈ@

dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/truedivdropout_2/cond/dropout/Floor*
T0*1
_output_shapes
:џџџџџџџџџЈ@
з
dropout_2/cond/Switch_1Switchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu

dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
T0*
N*3
_output_shapes!
:џџџџџџџџџЈ@: 
i
up_sampling2d_1/ShapeShapedropout_2/cond/Merge*
T0*
out_type0*
_output_shapes
:
m
#up_sampling2d_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Э
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape#up_sampling2d_1/strided_slice/stack%up_sampling2d_1/strided_slice/stack_1%up_sampling2d_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
f
up_sampling2d_1/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
u
up_sampling2d_1/mulMulup_sampling2d_1/strided_sliceup_sampling2d_1/Const*
T0*
_output_shapes
:
К
%up_sampling2d_1/ResizeNearestNeighborResizeNearestNeighbordropout_2/cond/Mergeup_sampling2d_1/mul*1
_output_shapes
:џџџџџџџџџИа@*
align_corners( *
T0
t
output/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @      
^
output/random_uniform/minConst*
valueB
 *8JЬН*
dtype0*
_output_shapes
: 
^
output/random_uniform/maxConst*
valueB
 *8JЬ=*
dtype0*
_output_shapes
: 
Ў
#output/random_uniform/RandomUniformRandomUniformoutput/random_uniform/shape*
seedБџх)*
T0*
dtype0*&
_output_shapes
:@*
seed2юЙ
w
output/random_uniform/subSuboutput/random_uniform/maxoutput/random_uniform/min*
_output_shapes
: *
T0

output/random_uniform/mulMul#output/random_uniform/RandomUniformoutput/random_uniform/sub*&
_output_shapes
:@*
T0

output/random_uniformAddoutput/random_uniform/muloutput/random_uniform/min*
T0*&
_output_shapes
:@

output/kernel
VariableV2*
shared_name *
dtype0*&
_output_shapes
:@*
	container *
shape:@
Р
output/kernel/AssignAssignoutput/kerneloutput/random_uniform* 
_class
loc:@output/kernel*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0

output/kernel/readIdentityoutput/kernel*
T0* 
_class
loc:@output/kernel*&
_output_shapes
:@
Y
output/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
w
output/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ѕ
output/bias/AssignAssignoutput/biasoutput/Const*
use_locking(*
T0*
_class
loc:@output/bias*
validate_shape(*
_output_shapes
:
n
output/bias/readIdentityoutput/bias*
T0*
_class
loc:@output/bias*
_output_shapes
:
q
 output/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

output/convolutionConv2D%up_sampling2d_1/ResizeNearestNeighboroutput/kernel/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа*
	dilations
*
T0

output/BiasAddBiasAddoutput/convolutionoutput/bias/read*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа*
T0
^
SGD/iterations/initial_valueConst*
dtype0	*
_output_shapes
: *
value	B	 R 
r
SGD/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
К
SGD/iterations/AssignAssignSGD/iterationsSGD/iterations/initial_value*
use_locking(*
T0	*!
_class
loc:@SGD/iterations*
validate_shape(*
_output_shapes
: 
s
SGD/iterations/readIdentitySGD/iterations*
T0	*!
_class
loc:@SGD/iterations*
_output_shapes
: 
Y
SGD/lr/initial_valueConst*
_output_shapes
: *
valueB
 *Зб8*
dtype0
j
SGD/lr
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 

SGD/lr/AssignAssignSGD/lrSGD/lr/initial_value*
use_locking(*
T0*
_class
loc:@SGD/lr*
validate_shape(*
_output_shapes
: 
[
SGD/lr/readIdentitySGD/lr*
_class
loc:@SGD/lr*
_output_shapes
: *
T0
_
SGD/momentum/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
p
SGD/momentum
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
В
SGD/momentum/AssignAssignSGD/momentumSGD/momentum/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@SGD/momentum*
validate_shape(
m
SGD/momentum/readIdentitySGD/momentum*
T0*
_class
loc:@SGD/momentum*
_output_shapes
: 
\
SGD/decay/initial_valueConst*
valueB
 *ЌХ'7*
dtype0*
_output_shapes
: 
m
	SGD/decay
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
І
SGD/decay/AssignAssign	SGD/decaySGD/decay/initial_value*
use_locking(*
T0*
_class
loc:@SGD/decay*
validate_shape(*
_output_shapes
: 
d
SGD/decay/readIdentity	SGD/decay*
_output_shapes
: *
T0*
_class
loc:@SGD/decay
Ж
output_targetPlaceholder*?
shape6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
output_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
v
loss/output_loss/subSuboutput/BiasAddoutput_target*
T0*1
_output_shapes
:џџџџџџџџџИа
m
loss/output_loss/AbsAbsloss/output_loss/sub*
T0*1
_output_shapes
:џџџџџџџџџИа
r
'loss/output_loss/Mean/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
Б
loss/output_loss/MeanMeanloss/output_loss/Abs'loss/output_loss/Mean/reduction_indices*-
_output_shapes
:џџџџџџџџџИа*
	keep_dims( *

Tidx0*
T0
z
)loss/output_loss/Mean_1/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB"      
Ќ
loss/output_loss/Mean_1Meanloss/output_loss/Mean)loss/output_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0
y
loss/output_loss/mulMulloss/output_loss/Mean_1output_sample_weights*#
_output_shapes
:џџџџџџџџџ*
T0
`
loss/output_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/output_loss/NotEqualNotEqualoutput_sample_weightsloss/output_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ

loss/output_loss/CastCastloss/output_loss/NotEqual*
Truncate( *#
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0

`
loss/output_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/output_loss/Mean_2Meanloss/output_loss/Castloss/output_loss/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

loss/output_loss/truedivRealDivloss/output_loss/mulloss/output_loss/Mean_2*
T0*#
_output_shapes
:џџџџџџџџџ
b
loss/output_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0

loss/output_loss/Mean_3Meanloss/output_loss/truedivloss/output_loss/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
U
loss/mulMul
loss/mul/xloss/output_loss/Mean_3*
_output_shapes
: *
T0
|
training/SGD/gradients/ShapeConst*
valueB *
_class
loc:@loss/mul*
dtype0*
_output_shapes
: 

 training/SGD/gradients/grad_ys_0Const*
valueB
 *  ?*
_class
loc:@loss/mul*
dtype0*
_output_shapes
: 
Г
training/SGD/gradients/FillFilltraining/SGD/gradients/Shape training/SGD/gradients/grad_ys_0*

index_type0*
_class
loc:@loss/mul*
_output_shapes
: *
T0
Ѓ
(training/SGD/gradients/loss/mul_grad/MulMultraining/SGD/gradients/Fillloss/output_loss/Mean_3*
_class
loc:@loss/mul*
_output_shapes
: *
T0

*training/SGD/gradients/loss/mul_grad/Mul_1Multraining/SGD/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
З
Atraining/SGD/gradients/loss/output_loss/Mean_3_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:**
_class 
loc:@loss/output_loss/Mean_3

;training/SGD/gradients/loss/output_loss/Mean_3_grad/ReshapeReshape*training/SGD/gradients/loss/mul_grad/Mul_1Atraining/SGD/gradients/loss/output_loss/Mean_3_grad/Reshape/shape*
T0*
Tshape0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
:
Н
9training/SGD/gradients/loss/output_loss/Mean_3_grad/ShapeShapeloss/output_loss/truediv*
T0*
out_type0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
:
Є
8training/SGD/gradients/loss/output_loss/Mean_3_grad/TileTile;training/SGD/gradients/loss/output_loss/Mean_3_grad/Reshape9training/SGD/gradients/loss/output_loss/Mean_3_grad/Shape*

Tmultiples0*
T0**
_class 
loc:@loss/output_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
П
;training/SGD/gradients/loss/output_loss/Mean_3_grad/Shape_1Shapeloss/output_loss/truediv*
T0*
out_type0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
:
Њ
;training/SGD/gradients/loss/output_loss/Mean_3_grad/Shape_2Const*
valueB **
_class 
loc:@loss/output_loss/Mean_3*
dtype0*
_output_shapes
: 
Џ
9training/SGD/gradients/loss/output_loss/Mean_3_grad/ConstConst*
valueB: **
_class 
loc:@loss/output_loss/Mean_3*
dtype0*
_output_shapes
:
Ђ
8training/SGD/gradients/loss/output_loss/Mean_3_grad/ProdProd;training/SGD/gradients/loss/output_loss/Mean_3_grad/Shape_19training/SGD/gradients/loss/output_loss/Mean_3_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/output_loss/Mean_3
Б
;training/SGD/gradients/loss/output_loss/Mean_3_grad/Const_1Const*
_output_shapes
:*
valueB: **
_class 
loc:@loss/output_loss/Mean_3*
dtype0
І
:training/SGD/gradients/loss/output_loss/Mean_3_grad/Prod_1Prod;training/SGD/gradients/loss/output_loss/Mean_3_grad/Shape_2;training/SGD/gradients/loss/output_loss/Mean_3_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/output_loss/Mean_3
Ћ
=training/SGD/gradients/loss/output_loss/Mean_3_grad/Maximum/yConst*
value	B :**
_class 
loc:@loss/output_loss/Mean_3*
dtype0*
_output_shapes
: 

;training/SGD/gradients/loss/output_loss/Mean_3_grad/MaximumMaximum:training/SGD/gradients/loss/output_loss/Mean_3_grad/Prod_1=training/SGD/gradients/loss/output_loss/Mean_3_grad/Maximum/y*
T0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
: 

<training/SGD/gradients/loss/output_loss/Mean_3_grad/floordivFloorDiv8training/SGD/gradients/loss/output_loss/Mean_3_grad/Prod;training/SGD/gradients/loss/output_loss/Mean_3_grad/Maximum*
_output_shapes
: *
T0**
_class 
loc:@loss/output_loss/Mean_3
ъ
8training/SGD/gradients/loss/output_loss/Mean_3_grad/CastCast<training/SGD/gradients/loss/output_loss/Mean_3_grad/floordiv*

SrcT0**
_class 
loc:@loss/output_loss/Mean_3*
Truncate( *
_output_shapes
: *

DstT0

;training/SGD/gradients/loss/output_loss/Mean_3_grad/truedivRealDiv8training/SGD/gradients/loss/output_loss/Mean_3_grad/Tile8training/SGD/gradients/loss/output_loss/Mean_3_grad/Cast*
T0**
_class 
loc:@loss/output_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Л
:training/SGD/gradients/loss/output_loss/truediv_grad/ShapeShapeloss/output_loss/mul*
T0*
out_type0*+
_class!
loc:@loss/output_loss/truediv*
_output_shapes
:
Ќ
<training/SGD/gradients/loss/output_loss/truediv_grad/Shape_1Const*
valueB *+
_class!
loc:@loss/output_loss/truediv*
dtype0*
_output_shapes
: 
Ч
Jtraining/SGD/gradients/loss/output_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs:training/SGD/gradients/loss/output_loss/truediv_grad/Shape<training/SGD/gradients/loss/output_loss/truediv_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*+
_class!
loc:@loss/output_loss/truediv
ј
<training/SGD/gradients/loss/output_loss/truediv_grad/RealDivRealDiv;training/SGD/gradients/loss/output_loss/Mean_3_grad/truedivloss/output_loss/Mean_2*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:џџџџџџџџџ
Ж
8training/SGD/gradients/loss/output_loss/truediv_grad/SumSum<training/SGD/gradients/loss/output_loss/truediv_grad/RealDivJtraining/SGD/gradients/loss/output_loss/truediv_grad/BroadcastGradientArgs*+
_class!
loc:@loss/output_loss/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
І
<training/SGD/gradients/loss/output_loss/truediv_grad/ReshapeReshape8training/SGD/gradients/loss/output_loss/truediv_grad/Sum:training/SGD/gradients/loss/output_loss/truediv_grad/Shape*
T0*
Tshape0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:џџџџџџџџџ
А
8training/SGD/gradients/loss/output_loss/truediv_grad/NegNegloss/output_loss/mul*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:џџџџџџџџџ
ї
>training/SGD/gradients/loss/output_loss/truediv_grad/RealDiv_1RealDiv8training/SGD/gradients/loss/output_loss/truediv_grad/Negloss/output_loss/Mean_2*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:џџџџџџџџџ
§
>training/SGD/gradients/loss/output_loss/truediv_grad/RealDiv_2RealDiv>training/SGD/gradients/loss/output_loss/truediv_grad/RealDiv_1loss/output_loss/Mean_2*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:џџџџџџџџџ

8training/SGD/gradients/loss/output_loss/truediv_grad/mulMul;training/SGD/gradients/loss/output_loss/Mean_3_grad/truediv>training/SGD/gradients/loss/output_loss/truediv_grad/RealDiv_2*#
_output_shapes
:џџџџџџџџџ*
T0*+
_class!
loc:@loss/output_loss/truediv
Ж
:training/SGD/gradients/loss/output_loss/truediv_grad/Sum_1Sum8training/SGD/gradients/loss/output_loss/truediv_grad/mulLtraining/SGD/gradients/loss/output_loss/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/output_loss/truediv

>training/SGD/gradients/loss/output_loss/truediv_grad/Reshape_1Reshape:training/SGD/gradients/loss/output_loss/truediv_grad/Sum_1<training/SGD/gradients/loss/output_loss/truediv_grad/Shape_1*
T0*
Tshape0*+
_class!
loc:@loss/output_loss/truediv*
_output_shapes
: 
Ж
6training/SGD/gradients/loss/output_loss/mul_grad/ShapeShapeloss/output_loss/Mean_1*
out_type0*'
_class
loc:@loss/output_loss/mul*
_output_shapes
:*
T0
Ж
8training/SGD/gradients/loss/output_loss/mul_grad/Shape_1Shapeoutput_sample_weights*
_output_shapes
:*
T0*
out_type0*'
_class
loc:@loss/output_loss/mul
З
Ftraining/SGD/gradients/loss/output_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6training/SGD/gradients/loss/output_loss/mul_grad/Shape8training/SGD/gradients/loss/output_loss/mul_grad/Shape_1*
T0*'
_class
loc:@loss/output_loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ч
4training/SGD/gradients/loss/output_loss/mul_grad/MulMul<training/SGD/gradients/loss/output_loss/truediv_grad/Reshapeoutput_sample_weights*
T0*'
_class
loc:@loss/output_loss/mul*#
_output_shapes
:џџџџџџџџџ
Ђ
4training/SGD/gradients/loss/output_loss/mul_grad/SumSum4training/SGD/gradients/loss/output_loss/mul_grad/MulFtraining/SGD/gradients/loss/output_loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*'
_class
loc:@loss/output_loss/mul*
_output_shapes
:

8training/SGD/gradients/loss/output_loss/mul_grad/ReshapeReshape4training/SGD/gradients/loss/output_loss/mul_grad/Sum6training/SGD/gradients/loss/output_loss/mul_grad/Shape*
T0*
Tshape0*'
_class
loc:@loss/output_loss/mul*#
_output_shapes
:џџџџџџџџџ
ы
6training/SGD/gradients/loss/output_loss/mul_grad/Mul_1Mulloss/output_loss/Mean_1<training/SGD/gradients/loss/output_loss/truediv_grad/Reshape*
T0*'
_class
loc:@loss/output_loss/mul*#
_output_shapes
:џџџџџџџџџ
Ј
6training/SGD/gradients/loss/output_loss/mul_grad/Sum_1Sum6training/SGD/gradients/loss/output_loss/mul_grad/Mul_1Htraining/SGD/gradients/loss/output_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*'
_class
loc:@loss/output_loss/mul

:training/SGD/gradients/loss/output_loss/mul_grad/Reshape_1Reshape6training/SGD/gradients/loss/output_loss/mul_grad/Sum_18training/SGD/gradients/loss/output_loss/mul_grad/Shape_1*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0*'
_class
loc:@loss/output_loss/mul
К
9training/SGD/gradients/loss/output_loss/Mean_1_grad/ShapeShapeloss/output_loss/Mean*
T0*
out_type0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:
І
8training/SGD/gradients/loss/output_loss/Mean_1_grad/SizeConst*
value	B :**
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
: 
є
7training/SGD/gradients/loss/output_loss/Mean_1_grad/addAdd)loss/output_loss/Mean_1/reduction_indices8training/SGD/gradients/loss/output_loss/Mean_1_grad/Size*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:

7training/SGD/gradients/loss/output_loss/Mean_1_grad/modFloorMod7training/SGD/gradients/loss/output_loss/Mean_1_grad/add8training/SGD/gradients/loss/output_loss/Mean_1_grad/Size*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:
Б
;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_1Const*
valueB:**
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
:
­
?training/SGD/gradients/loss/output_loss/Mean_1_grad/range/startConst*
value	B : **
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
: 
­
?training/SGD/gradients/loss/output_loss/Mean_1_grad/range/deltaConst*
value	B :**
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
: 
в
9training/SGD/gradients/loss/output_loss/Mean_1_grad/rangeRange?training/SGD/gradients/loss/output_loss/Mean_1_grad/range/start8training/SGD/gradients/loss/output_loss/Mean_1_grad/Size?training/SGD/gradients/loss/output_loss/Mean_1_grad/range/delta**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:*

Tidx0
Ќ
>training/SGD/gradients/loss/output_loss/Mean_1_grad/Fill/valueConst*
value	B :**
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
: 
 
8training/SGD/gradients/loss/output_loss/Mean_1_grad/FillFill;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_1>training/SGD/gradients/loss/output_loss/Mean_1_grad/Fill/value*
T0*

index_type0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:

Atraining/SGD/gradients/loss/output_loss/Mean_1_grad/DynamicStitchDynamicStitch9training/SGD/gradients/loss/output_loss/Mean_1_grad/range7training/SGD/gradients/loss/output_loss/Mean_1_grad/mod9training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape8training/SGD/gradients/loss/output_loss/Mean_1_grad/Fill*
T0**
_class 
loc:@loss/output_loss/Mean_1*
N*
_output_shapes
:
Ћ
=training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
value	B :**
_class 
loc:@loss/output_loss/Mean_1*
dtype0

;training/SGD/gradients/loss/output_loss/Mean_1_grad/MaximumMaximumAtraining/SGD/gradients/loss/output_loss/Mean_1_grad/DynamicStitch=training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum/y*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:

<training/SGD/gradients/loss/output_loss/Mean_1_grad/floordivFloorDiv9training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape;training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:
Х
;training/SGD/gradients/loss/output_loss/Mean_1_grad/ReshapeReshape8training/SGD/gradients/loss/output_loss/mul_grad/ReshapeAtraining/SGD/gradients/loss/output_loss/Mean_1_grad/DynamicStitch*
T0*
Tshape0**
_class 
loc:@loss/output_loss/Mean_1*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
С
8training/SGD/gradients/loss/output_loss/Mean_1_grad/TileTile;training/SGD/gradients/loss/output_loss/Mean_1_grad/Reshape<training/SGD/gradients/loss/output_loss/Mean_1_grad/floordiv*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*

Tmultiples0*
T0**
_class 
loc:@loss/output_loss/Mean_1
М
;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_2Shapeloss/output_loss/Mean*
T0*
out_type0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:
О
;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_3Shapeloss/output_loss/Mean_1*
T0*
out_type0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:
Џ
9training/SGD/gradients/loss/output_loss/Mean_1_grad/ConstConst*
valueB: **
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
:
Ђ
8training/SGD/gradients/loss/output_loss/Mean_1_grad/ProdProd;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_29training/SGD/gradients/loss/output_loss/Mean_1_grad/Const*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
Б
;training/SGD/gradients/loss/output_loss/Mean_1_grad/Const_1Const*
valueB: **
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
:
І
:training/SGD/gradients/loss/output_loss/Mean_1_grad/Prod_1Prod;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_3;training/SGD/gradients/loss/output_loss/Mean_1_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/output_loss/Mean_1
­
?training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum_1/yConst*
value	B :**
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
: 

=training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum_1Maximum:training/SGD/gradients/loss/output_loss/Mean_1_grad/Prod_1?training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum_1/y*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: 

>training/SGD/gradients/loss/output_loss/Mean_1_grad/floordiv_1FloorDiv8training/SGD/gradients/loss/output_loss/Mean_1_grad/Prod=training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum_1*
_output_shapes
: *
T0**
_class 
loc:@loss/output_loss/Mean_1
ь
8training/SGD/gradients/loss/output_loss/Mean_1_grad/CastCast>training/SGD/gradients/loss/output_loss/Mean_1_grad/floordiv_1*

SrcT0**
_class 
loc:@loss/output_loss/Mean_1*
Truncate( *
_output_shapes
: *

DstT0

;training/SGD/gradients/loss/output_loss/Mean_1_grad/truedivRealDiv8training/SGD/gradients/loss/output_loss/Mean_1_grad/Tile8training/SGD/gradients/loss/output_loss/Mean_1_grad/Cast*-
_output_shapes
:џџџџџџџџџИа*
T0**
_class 
loc:@loss/output_loss/Mean_1
Е
7training/SGD/gradients/loss/output_loss/Mean_grad/ShapeShapeloss/output_loss/Abs*
T0*
out_type0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:
Ђ
6training/SGD/gradients/loss/output_loss/Mean_grad/SizeConst*
value	B :*(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 
ш
5training/SGD/gradients/loss/output_loss/Mean_grad/addAdd'loss/output_loss/Mean/reduction_indices6training/SGD/gradients/loss/output_loss/Mean_grad/Size*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
ћ
5training/SGD/gradients/loss/output_loss/Mean_grad/modFloorMod5training/SGD/gradients/loss/output_loss/Mean_grad/add6training/SGD/gradients/loss/output_loss/Mean_grad/Size*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
І
9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_1Const*
valueB *(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 
Љ
=training/SGD/gradients/loss/output_loss/Mean_grad/range/startConst*
value	B : *(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 
Љ
=training/SGD/gradients/loss/output_loss/Mean_grad/range/deltaConst*
value	B :*(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 
Ш
7training/SGD/gradients/loss/output_loss/Mean_grad/rangeRange=training/SGD/gradients/loss/output_loss/Mean_grad/range/start6training/SGD/gradients/loss/output_loss/Mean_grad/Size=training/SGD/gradients/loss/output_loss/Mean_grad/range/delta*

Tidx0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:
Ј
<training/SGD/gradients/loss/output_loss/Mean_grad/Fill/valueConst*
value	B :*(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 

6training/SGD/gradients/loss/output_loss/Mean_grad/FillFill9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_1<training/SGD/gradients/loss/output_loss/Mean_grad/Fill/value*
T0*

index_type0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 

?training/SGD/gradients/loss/output_loss/Mean_grad/DynamicStitchDynamicStitch7training/SGD/gradients/loss/output_loss/Mean_grad/range5training/SGD/gradients/loss/output_loss/Mean_grad/mod7training/SGD/gradients/loss/output_loss/Mean_grad/Shape6training/SGD/gradients/loss/output_loss/Mean_grad/Fill*(
_class
loc:@loss/output_loss/Mean*
N*
_output_shapes
:*
T0
Ї
;training/SGD/gradients/loss/output_loss/Mean_grad/Maximum/yConst*
value	B :*(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 

9training/SGD/gradients/loss/output_loss/Mean_grad/MaximumMaximum?training/SGD/gradients/loss/output_loss/Mean_grad/DynamicStitch;training/SGD/gradients/loss/output_loss/Mean_grad/Maximum/y*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:

:training/SGD/gradients/loss/output_loss/Mean_grad/floordivFloorDiv7training/SGD/gradients/loss/output_loss/Mean_grad/Shape9training/SGD/gradients/loss/output_loss/Mean_grad/Maximum*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:
Я
9training/SGD/gradients/loss/output_loss/Mean_grad/ReshapeReshape;training/SGD/gradients/loss/output_loss/Mean_1_grad/truediv?training/SGD/gradients/loss/output_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*(
_class
loc:@loss/output_loss/Mean*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ц
6training/SGD/gradients/loss/output_loss/Mean_grad/TileTile9training/SGD/gradients/loss/output_loss/Mean_grad/Reshape:training/SGD/gradients/loss/output_loss/Mean_grad/floordiv*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*

Tmultiples0*
T0*(
_class
loc:@loss/output_loss/Mean
З
9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_2Shapeloss/output_loss/Abs*
T0*
out_type0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:
И
9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_3Shapeloss/output_loss/Mean*
_output_shapes
:*
T0*
out_type0*(
_class
loc:@loss/output_loss/Mean
Ћ
7training/SGD/gradients/loss/output_loss/Mean_grad/ConstConst*
valueB: *(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
:

6training/SGD/gradients/loss/output_loss/Mean_grad/ProdProd9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_27training/SGD/gradients/loss/output_loss/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
­
9training/SGD/gradients/loss/output_loss/Mean_grad/Const_1Const*
valueB: *(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
:

8training/SGD/gradients/loss/output_loss/Mean_grad/Prod_1Prod9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_39training/SGD/gradients/loss/output_loss/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
Љ
=training/SGD/gradients/loss/output_loss/Mean_grad/Maximum_1/yConst*
_output_shapes
: *
value	B :*(
_class
loc:@loss/output_loss/Mean*
dtype0

;training/SGD/gradients/loss/output_loss/Mean_grad/Maximum_1Maximum8training/SGD/gradients/loss/output_loss/Mean_grad/Prod_1=training/SGD/gradients/loss/output_loss/Mean_grad/Maximum_1/y*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 

<training/SGD/gradients/loss/output_loss/Mean_grad/floordiv_1FloorDiv6training/SGD/gradients/loss/output_loss/Mean_grad/Prod;training/SGD/gradients/loss/output_loss/Mean_grad/Maximum_1*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
ц
6training/SGD/gradients/loss/output_loss/Mean_grad/CastCast<training/SGD/gradients/loss/output_loss/Mean_grad/floordiv_1*

SrcT0*(
_class
loc:@loss/output_loss/Mean*
Truncate( *
_output_shapes
: *

DstT0

9training/SGD/gradients/loss/output_loss/Mean_grad/truedivRealDiv6training/SGD/gradients/loss/output_loss/Mean_grad/Tile6training/SGD/gradients/loss/output_loss/Mean_grad/Cast*
T0*(
_class
loc:@loss/output_loss/Mean*1
_output_shapes
:џџџџџџџџџИа
И
5training/SGD/gradients/loss/output_loss/Abs_grad/SignSignloss/output_loss/sub*
T0*'
_class
loc:@loss/output_loss/Abs*1
_output_shapes
:џџџџџџџџџИа

4training/SGD/gradients/loss/output_loss/Abs_grad/mulMul9training/SGD/gradients/loss/output_loss/Mean_grad/truediv5training/SGD/gradients/loss/output_loss/Abs_grad/Sign*1
_output_shapes
:џџџџџџџџџИа*
T0*'
_class
loc:@loss/output_loss/Abs
­
6training/SGD/gradients/loss/output_loss/sub_grad/ShapeShapeoutput/BiasAdd*
T0*
out_type0*'
_class
loc:@loss/output_loss/sub*
_output_shapes
:
Ў
8training/SGD/gradients/loss/output_loss/sub_grad/Shape_1Shapeoutput_target*
_output_shapes
:*
T0*
out_type0*'
_class
loc:@loss/output_loss/sub
З
Ftraining/SGD/gradients/loss/output_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs6training/SGD/gradients/loss/output_loss/sub_grad/Shape8training/SGD/gradients/loss/output_loss/sub_grad/Shape_1*
T0*'
_class
loc:@loss/output_loss/sub*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ђ
4training/SGD/gradients/loss/output_loss/sub_grad/SumSum4training/SGD/gradients/loss/output_loss/Abs_grad/mulFtraining/SGD/gradients/loss/output_loss/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*'
_class
loc:@loss/output_loss/sub*
_output_shapes
:
Є
8training/SGD/gradients/loss/output_loss/sub_grad/ReshapeReshape4training/SGD/gradients/loss/output_loss/sub_grad/Sum6training/SGD/gradients/loss/output_loss/sub_grad/Shape*
T0*
Tshape0*'
_class
loc:@loss/output_loss/sub*1
_output_shapes
:џџџџџџџџџИа
І
6training/SGD/gradients/loss/output_loss/sub_grad/Sum_1Sum4training/SGD/gradients/loss/output_loss/Abs_grad/mulHtraining/SGD/gradients/loss/output_loss/sub_grad/BroadcastGradientArgs:1*'
_class
loc:@loss/output_loss/sub*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
П
4training/SGD/gradients/loss/output_loss/sub_grad/NegNeg6training/SGD/gradients/loss/output_loss/sub_grad/Sum_1*'
_class
loc:@loss/output_loss/sub*
_output_shapes
:*
T0
С
:training/SGD/gradients/loss/output_loss/sub_grad/Reshape_1Reshape4training/SGD/gradients/loss/output_loss/sub_grad/Neg8training/SGD/gradients/loss/output_loss/sub_grad/Shape_1*
T0*
Tshape0*'
_class
loc:@loss/output_loss/sub*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
о
6training/SGD/gradients/output/BiasAdd_grad/BiasAddGradBiasAddGrad8training/SGD/gradients/loss/output_loss/sub_grad/Reshape*
data_formatNHWC*
_output_shapes
:*
T0*!
_class
loc:@output/BiasAdd
х
5training/SGD/gradients/output/convolution_grad/ShapeNShapeN%up_sampling2d_1/ResizeNearestNeighboroutput/kernel/read*
T0*
out_type0*%
_class
loc:@output/convolution*
N* 
_output_shapes
::
Џ
Btraining/SGD/gradients/output/convolution_grad/Conv2DBackpropInputConv2DBackpropInput5training/SGD/gradients/output/convolution_grad/ShapeNoutput/kernel/read8training/SGD/gradients/loss/output_loss/sub_grad/Reshape*1
_output_shapes
:џџџџџџџџџИа@*
	dilations
*
T0*%
_class
loc:@output/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Л
Ctraining/SGD/gradients/output/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_1/ResizeNearestNeighbor7training/SGD/gradients/output/convolution_grad/ShapeN:18training/SGD/gradients/loss/output_loss/sub_grad/Reshape*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@*
	dilations
*
T0*%
_class
loc:@output/convolution*
strides
*
data_formatNHWC
ы
`training/SGD/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"  (  *8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*
dtype0*
_output_shapes
:
Љ
[training/SGD/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradBtraining/SGD/gradients/output/convolution_grad/Conv2DBackpropInput`training/SGD/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*1
_output_shapes
:џџџџџџџџџЈ@*
align_corners( *
T0
Ь
:training/SGD/gradients/dropout_2/cond/Merge_grad/cond_gradSwitch[training/SGD/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGraddropout_2/cond/pred_id*
T0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@
н
training/SGD/gradients/SwitchSwitchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@
Д
training/SGD/gradients/IdentityIdentitytraining/SGD/gradients/Switch:1*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*1
_output_shapes
:џџџџџџџџџЈ@
Љ
training/SGD/gradients/Shape_1Shapetraining/SGD/gradients/Switch:1*
out_type0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
_output_shapes
:*
T0
Е
"training/SGD/gradients/zeros/ConstConst ^training/SGD/gradients/Identity*
valueB
 *    **
_class 
loc:@leaky_re_lu_2/LeakyRelu*
dtype0*
_output_shapes
: 
т
training/SGD/gradients/zerosFilltraining/SGD/gradients/Shape_1"training/SGD/gradients/zeros/Const*
T0*

index_type0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*1
_output_shapes
:џџџџџџџџџЈ@

=training/SGD/gradients/dropout_2/cond/Switch_1_grad/cond_gradMerge:training/SGD/gradients/dropout_2/cond/Merge_grad/cond_gradtraining/SGD/gradients/zeros*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџЈ@: 
Щ
<training/SGD/gradients/dropout_2/cond/dropout/mul_grad/ShapeShapedropout_2/cond/dropout/truediv*
T0*
out_type0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:
Щ
>training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Shape_1Shapedropout_2/cond/dropout/Floor*
T0*
out_type0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:
Я
Ltraining/SGD/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Shape>training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul

:training/SGD/gradients/dropout_2/cond/dropout/mul_grad/MulMul<training/SGD/gradients/dropout_2/cond/Merge_grad/cond_grad:1dropout_2/cond/dropout/Floor*-
_class#
!loc:@dropout_2/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџЈ@*
T0
К
:training/SGD/gradients/dropout_2/cond/dropout/mul_grad/SumSum:training/SGD/gradients/dropout_2/cond/dropout/mul_grad/MulLtraining/SGD/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
М
>training/SGD/gradients/dropout_2/cond/dropout/mul_grad/ReshapeReshape:training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Sum<training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Shape*1
_output_shapes
:џџџџџџџџџЈ@*
T0*
Tshape0*-
_class#
!loc:@dropout_2/cond/dropout/mul

<training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Muldropout_2/cond/dropout/truediv<training/SGD/gradients/dropout_2/cond/Merge_grad/cond_grad:1*1
_output_shapes
:џџџџџџџџџЈ@*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
Р
<training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Sum_1Sum<training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Ntraining/SGD/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Т
@training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape<training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Sum_1>training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*-
_class#
!loc:@dropout_2/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџЈ@
Х
@training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/ShapeShapedropout_2/cond/mul*
T0*
out_type0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
_output_shapes
:
И
Btraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Shape_1Const*
_output_shapes
: *
valueB *1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
dtype0
п
Ptraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs@training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/ShapeBtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Btraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/RealDivRealDiv>training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Reshapedropout_2/cond/dropout/sub*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@*
T0
Ю
>training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/SumSumBtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/RealDivPtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/BroadcastGradientArgs*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ь
Btraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/ReshapeReshape>training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Sum@training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Shape*
Tshape0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@*
T0
Ш
>training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/NegNegdropout_2/cond/mul*1
_output_shapes
:џџџџџџџџџЈ@*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv

Dtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_1RealDiv>training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Negdropout_2/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџЈ@*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv
 
Dtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_2RealDivDtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_1dropout_2/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџЈ@*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv
К
>training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/mulMul>training/SGD/gradients/dropout_2/cond/dropout/mul_grad/ReshapeDtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_2*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@*
T0
Ю
@training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Sum_1Sum>training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/mulRtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv
З
Dtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Reshape_1Reshape@training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Sum_1Btraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Shape_1*
T0*
Tshape0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
_output_shapes
: 
Ж
4training/SGD/gradients/dropout_2/cond/mul_grad/ShapeShapedropout_2/cond/mul/Switch:1*
T0*
out_type0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:
 
6training/SGD/gradients/dropout_2/cond/mul_grad/Shape_1Const*
valueB *%
_class
loc:@dropout_2/cond/mul*
dtype0*
_output_shapes
: 
Џ
Dtraining/SGD/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4training/SGD/gradients/dropout_2/cond/mul_grad/Shape6training/SGD/gradients/dropout_2/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_2/cond/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
і
2training/SGD/gradients/dropout_2/cond/mul_grad/MulMulBtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Reshapedropout_2/cond/mul/y*
T0*%
_class
loc:@dropout_2/cond/mul*1
_output_shapes
:џџџџџџџџџЈ@

2training/SGD/gradients/dropout_2/cond/mul_grad/SumSum2training/SGD/gradients/dropout_2/cond/mul_grad/MulDtraining/SGD/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:

6training/SGD/gradients/dropout_2/cond/mul_grad/ReshapeReshape2training/SGD/gradients/dropout_2/cond/mul_grad/Sum4training/SGD/gradients/dropout_2/cond/mul_grad/Shape*
T0*
Tshape0*%
_class
loc:@dropout_2/cond/mul*1
_output_shapes
:џџџџџџџџџЈ@
џ
4training/SGD/gradients/dropout_2/cond/mul_grad/Mul_1Muldropout_2/cond/mul/Switch:1Btraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Reshape*
T0*%
_class
loc:@dropout_2/cond/mul*1
_output_shapes
:џџџџџџџџџЈ@
 
4training/SGD/gradients/dropout_2/cond/mul_grad/Sum_1Sum4training/SGD/gradients/dropout_2/cond/mul_grad/Mul_1Ftraining/SGD/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:

8training/SGD/gradients/dropout_2/cond/mul_grad/Reshape_1Reshape4training/SGD/gradients/dropout_2/cond/mul_grad/Sum_16training/SGD/gradients/dropout_2/cond/mul_grad/Shape_1*
T0*
Tshape0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
: 
п
training/SGD/gradients/Switch_1Switchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@
Ж
!training/SGD/gradients/Identity_1Identitytraining/SGD/gradients/Switch_1*1
_output_shapes
:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu
Љ
training/SGD/gradients/Shape_2Shapetraining/SGD/gradients/Switch_1*
T0*
out_type0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
_output_shapes
:
Й
$training/SGD/gradients/zeros_1/ConstConst"^training/SGD/gradients/Identity_1*
valueB
 *    **
_class 
loc:@leaky_re_lu_2/LeakyRelu*
dtype0*
_output_shapes
: 
ц
training/SGD/gradients/zeros_1Filltraining/SGD/gradients/Shape_2$training/SGD/gradients/zeros_1/Const*
T0*

index_type0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*1
_output_shapes
:џџџџџџџџџЈ@

?training/SGD/gradients/dropout_2/cond/mul/Switch_grad/cond_gradMergetraining/SGD/gradients/zeros_16training/SGD/gradients/dropout_2/cond/mul_grad/Reshape*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџЈ@: 

training/SGD/gradients/AddNAddN=training/SGD/gradients/dropout_2/cond/Switch_1_grad/cond_grad?training/SGD/gradients/dropout_2/cond/mul/Switch_grad/cond_grad*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N*1
_output_shapes
:џџџџџџџџџЈ@
љ
Atraining/SGD/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGradLeakyReluGradtraining/SGD/gradients/AddNconv2d_1/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
alpha%>*1
_output_shapes
:џџџџџџџџџЈ@
ы
8training/SGD/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGradAtraining/SGD/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_1/BiasAdd*
data_formatNHWC*
_output_shapes
:@
н
7training/SGD/gradients/conv2d_1/convolution_grad/ShapeNShapeNmax_pooling2d_1/MaxPoolconv2d_1/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_1/convolution*
N* 
_output_shapes
::
Р
Dtraining/SGD/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput7training/SGD/gradients/conv2d_1/convolution_grad/ShapeNconv2d_1/kernel/readAtraining/SGD/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ 
М
Etraining/SGD/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_1/MaxPool9training/SGD/gradients/conv2d_1/convolution_grad/ShapeN:1Atraining/SGD/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
paddingSAME*&
_output_shapes
: @*
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
§
?training/SGD/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGraddropout_1/cond/Mergemax_pooling2d_1/MaxPoolDtraining/SGD/gradients/conv2d_1/convolution_grad/Conv2DBackpropInput*
T0**
_class 
loc:@max_pooling2d_1/MaxPool*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа 
Ђ
:training/SGD/gradients/dropout_1/cond/Merge_grad/cond_gradSwitch?training/SGD/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGraddropout_1/cond/pred_id*
T0**
_class 
loc:@max_pooling2d_1/MaxPool*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа 
п
training/SGD/gradients/Switch_2Switchleaky_re_lu_1/LeakyReludropout_1/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа 
И
!training/SGD/gradients/Identity_2Identity!training/SGD/gradients/Switch_2:1*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*1
_output_shapes
:џџџџџџџџџИа 
Ћ
training/SGD/gradients/Shape_3Shape!training/SGD/gradients/Switch_2:1*
out_type0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
_output_shapes
:*
T0
Й
$training/SGD/gradients/zeros_2/ConstConst"^training/SGD/gradients/Identity_2*
valueB
 *    **
_class 
loc:@leaky_re_lu_1/LeakyRelu*
dtype0*
_output_shapes
: 
ц
training/SGD/gradients/zeros_2Filltraining/SGD/gradients/Shape_3$training/SGD/gradients/zeros_2/Const*
T0*

index_type0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*1
_output_shapes
:џџџџџџџџџИа 

=training/SGD/gradients/dropout_1/cond/Switch_1_grad/cond_gradMerge:training/SGD/gradients/dropout_1/cond/Merge_grad/cond_gradtraining/SGD/gradients/zeros_2*3
_output_shapes!
:џџџџџџџџџИа : *
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
N
Щ
<training/SGD/gradients/dropout_1/cond/dropout/mul_grad/ShapeShapedropout_1/cond/dropout/truediv*
T0*
out_type0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:
Щ
>training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Shape_1Shapedropout_1/cond/dropout/Floor*
T0*
out_type0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:
Я
Ltraining/SGD/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Shape>training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*-
_class#
!loc:@dropout_1/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

:training/SGD/gradients/dropout_1/cond/dropout/mul_grad/MulMul<training/SGD/gradients/dropout_1/cond/Merge_grad/cond_grad:1dropout_1/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџИа 
К
:training/SGD/gradients/dropout_1/cond/dropout/mul_grad/SumSum:training/SGD/gradients/dropout_1/cond/dropout/mul_grad/MulLtraining/SGD/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
М
>training/SGD/gradients/dropout_1/cond/dropout/mul_grad/ReshapeReshape:training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Sum<training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Shape*1
_output_shapes
:џџџџџџџџџИа *
T0*
Tshape0*-
_class#
!loc:@dropout_1/cond/dropout/mul

<training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Muldropout_1/cond/dropout/truediv<training/SGD/gradients/dropout_1/cond/Merge_grad/cond_grad:1*-
_class#
!loc:@dropout_1/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџИа *
T0
Р
<training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Sum<training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Ntraining/SGD/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
Т
@training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape<training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Sum_1>training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*-
_class#
!loc:@dropout_1/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџИа 
Х
@training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/ShapeShapedropout_1/cond/mul*
T0*
out_type0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
:
И
Btraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1Const*
valueB *1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
dtype0*
_output_shapes
: 
п
Ptraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs@training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/ShapeBtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv

Btraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/RealDivRealDiv>training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Reshapedropout_1/cond/dropout/sub*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа *
T0
Ю
>training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/SumSumBtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/RealDivPtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
:
Ь
Btraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/ReshapeReshape>training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Sum@training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Shape*
T0*
Tshape0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа 
Ш
>training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/NegNegdropout_1/cond/mul*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа 

Dtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1RealDiv>training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Negdropout_1/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџИа *
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
 
Dtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2RealDivDtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1dropout_1/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџИа *
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
К
>training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/mulMul>training/SGD/gradients/dropout_1/cond/dropout/mul_grad/ReshapeDtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа 
Ю
@training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1Sum>training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/mulRtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
:
З
Dtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Reshape_1Reshape@training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1Btraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
Ж
4training/SGD/gradients/dropout_1/cond/mul_grad/ShapeShapedropout_1/cond/mul/Switch:1*
T0*
out_type0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:
 
6training/SGD/gradients/dropout_1/cond/mul_grad/Shape_1Const*
valueB *%
_class
loc:@dropout_1/cond/mul*
dtype0*
_output_shapes
: 
Џ
Dtraining/SGD/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4training/SGD/gradients/dropout_1/cond/mul_grad/Shape6training/SGD/gradients/dropout_1/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_1/cond/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
і
2training/SGD/gradients/dropout_1/cond/mul_grad/MulMulBtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Reshapedropout_1/cond/mul/y*
T0*%
_class
loc:@dropout_1/cond/mul*1
_output_shapes
:џџџџџџџџџИа 

2training/SGD/gradients/dropout_1/cond/mul_grad/SumSum2training/SGD/gradients/dropout_1/cond/mul_grad/MulDtraining/SGD/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

6training/SGD/gradients/dropout_1/cond/mul_grad/ReshapeReshape2training/SGD/gradients/dropout_1/cond/mul_grad/Sum4training/SGD/gradients/dropout_1/cond/mul_grad/Shape*
T0*
Tshape0*%
_class
loc:@dropout_1/cond/mul*1
_output_shapes
:џџџџџџџџџИа 
џ
4training/SGD/gradients/dropout_1/cond/mul_grad/Mul_1Muldropout_1/cond/mul/Switch:1Btraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Reshape*
T0*%
_class
loc:@dropout_1/cond/mul*1
_output_shapes
:џџџџџџџџџИа 
 
4training/SGD/gradients/dropout_1/cond/mul_grad/Sum_1Sum4training/SGD/gradients/dropout_1/cond/mul_grad/Mul_1Ftraining/SGD/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*%
_class
loc:@dropout_1/cond/mul

8training/SGD/gradients/dropout_1/cond/mul_grad/Reshape_1Reshape4training/SGD/gradients/dropout_1/cond/mul_grad/Sum_16training/SGD/gradients/dropout_1/cond/mul_grad/Shape_1*
T0*
Tshape0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
: 
п
training/SGD/gradients/Switch_3Switchleaky_re_lu_1/LeakyReludropout_1/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа 
Ж
!training/SGD/gradients/Identity_3Identitytraining/SGD/gradients/Switch_3*1
_output_shapes
:џџџџџџџџџИа *
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu
Љ
training/SGD/gradients/Shape_4Shapetraining/SGD/gradients/Switch_3*
T0*
out_type0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
_output_shapes
:
Й
$training/SGD/gradients/zeros_3/ConstConst"^training/SGD/gradients/Identity_3*
_output_shapes
: *
valueB
 *    **
_class 
loc:@leaky_re_lu_1/LeakyRelu*
dtype0
ц
training/SGD/gradients/zeros_3Filltraining/SGD/gradients/Shape_4$training/SGD/gradients/zeros_3/Const*
T0*

index_type0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*1
_output_shapes
:џџџџџџџџџИа 

?training/SGD/gradients/dropout_1/cond/mul/Switch_grad/cond_gradMergetraining/SGD/gradients/zeros_36training/SGD/gradients/dropout_1/cond/mul_grad/Reshape**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџИа : *
T0

training/SGD/gradients/AddN_1AddN=training/SGD/gradients/dropout_1/cond/Switch_1_grad/cond_grad?training/SGD/gradients/dropout_1/cond/mul/Switch_grad/cond_grad*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
N*1
_output_shapes
:џџџџџџџџџИа 
ј
Atraining/SGD/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGradLeakyReluGradtraining/SGD/gradients/AddN_1input/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
alpha%>*1
_output_shapes
:џџџџџџџџџИа 
х
5training/SGD/gradients/input/BiasAdd_grad/BiasAddGradBiasAddGradAtraining/SGD/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad* 
_class
loc:@input/BiasAdd*
data_formatNHWC*
_output_shapes
: *
T0
Ш
4training/SGD/gradients/input/convolution_grad/ShapeNShapeNinput_inputinput/kernel/read*
T0*
out_type0*$
_class
loc:@input/convolution*
N* 
_output_shapes
::
Д
Atraining/SGD/gradients/input/convolution_grad/Conv2DBackpropInputConv2DBackpropInput4training/SGD/gradients/input/convolution_grad/ShapeNinput/kernel/readAtraining/SGD/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа*
	dilations
*
T0*$
_class
loc:@input/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ї
Btraining/SGD/gradients/input/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterinput_input6training/SGD/gradients/input/convolution_grad/ShapeN:1Atraining/SGD/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0*$
_class
loc:@input/convolution
^
training/SGD/AssignAdd/valueConst*
_output_shapes
: *
value	B	 R*
dtype0	
Ј
training/SGD/AssignAdd	AssignAddSGD/iterationstraining/SGD/AssignAdd/value*
use_locking( *
T0	*!
_class
loc:@SGD/iterations*
_output_shapes
: 
n
training/SGD/CastCastSGD/iterations/read*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
[
training/SGD/mulMulSGD/decay/readtraining/SGD/Cast*
_output_shapes
: *
T0
W
training/SGD/add/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
^
training/SGD/addAddtraining/SGD/add/xtraining/SGD/mul*
_output_shapes
: *
T0
[
training/SGD/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
training/SGD/truedivRealDivtraining/SGD/truediv/xtraining/SGD/add*
T0*
_output_shapes
: 
]
training/SGD/mul_1MulSGD/lr/readtraining/SGD/truediv*
T0*
_output_shapes
: 
w
training/SGD/zerosConst*
dtype0*&
_output_shapes
: *%
valueB *    

training/SGD/Variable
VariableV2*
shared_name *
dtype0*&
_output_shapes
: *
	container *
shape: 
е
training/SGD/Variable/AssignAssigntraining/SGD/Variabletraining/SGD/zeros*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@training/SGD/Variable

training/SGD/Variable/readIdentitytraining/SGD/Variable*
T0*(
_class
loc:@training/SGD/Variable*&
_output_shapes
: 
a
training/SGD/zeros_1Const*
dtype0*
_output_shapes
: *
valueB *    

training/SGD/Variable_1
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
б
training/SGD/Variable_1/AssignAssigntraining/SGD/Variable_1training/SGD/zeros_1*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_1*
validate_shape(*
_output_shapes
: 

training/SGD/Variable_1/readIdentitytraining/SGD/Variable_1**
_class 
loc:@training/SGD/Variable_1*
_output_shapes
: *
T0
}
$training/SGD/zeros_2/shape_as_tensorConst*
_output_shapes
:*%
valueB"          @   *
dtype0
_
training/SGD/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ё
training/SGD/zeros_2Fill$training/SGD/zeros_2/shape_as_tensortraining/SGD/zeros_2/Const*&
_output_shapes
: @*
T0*

index_type0

training/SGD/Variable_2
VariableV2*
shape: @*
shared_name *
dtype0*&
_output_shapes
: @*
	container 
н
training/SGD/Variable_2/AssignAssigntraining/SGD/Variable_2training/SGD/zeros_2*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_2*
validate_shape(*&
_output_shapes
: @

training/SGD/Variable_2/readIdentitytraining/SGD/Variable_2*
T0**
_class 
loc:@training/SGD/Variable_2*&
_output_shapes
: @
a
training/SGD/zeros_3Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/SGD/Variable_3
VariableV2*
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
б
training/SGD/Variable_3/AssignAssigntraining/SGD/Variable_3training/SGD/zeros_3*
_output_shapes
:@*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_3*
validate_shape(

training/SGD/Variable_3/readIdentitytraining/SGD/Variable_3*
T0**
_class 
loc:@training/SGD/Variable_3*
_output_shapes
:@
}
$training/SGD/zeros_4/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @      
_
training/SGD/zeros_4/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ё
training/SGD/zeros_4Fill$training/SGD/zeros_4/shape_as_tensortraining/SGD/zeros_4/Const*
T0*

index_type0*&
_output_shapes
:@

training/SGD/Variable_4
VariableV2*&
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
н
training/SGD/Variable_4/AssignAssigntraining/SGD/Variable_4training/SGD/zeros_4*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_4*
validate_shape(*&
_output_shapes
:@

training/SGD/Variable_4/readIdentitytraining/SGD/Variable_4*
T0**
_class 
loc:@training/SGD/Variable_4*&
_output_shapes
:@
a
training/SGD/zeros_5Const*
_output_shapes
:*
valueB*    *
dtype0

training/SGD/Variable_5
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
б
training/SGD/Variable_5/AssignAssigntraining/SGD/Variable_5training/SGD/zeros_5*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_5*
validate_shape(*
_output_shapes
:

training/SGD/Variable_5/readIdentitytraining/SGD/Variable_5**
_class 
loc:@training/SGD/Variable_5*
_output_shapes
:*
T0
y
training/SGD/mul_2MulSGD/momentum/readtraining/SGD/Variable/read*
T0*&
_output_shapes
: 
Ђ
training/SGD/mul_3Multraining/SGD/mul_1Btraining/SGD/gradients/input/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
p
training/SGD/subSubtraining/SGD/mul_2training/SGD/mul_3*&
_output_shapes
: *
T0
Ъ
training/SGD/AssignAssigntraining/SGD/Variabletraining/SGD/sub*
use_locking(*
T0*(
_class
loc:@training/SGD/Variable*
validate_shape(*&
_output_shapes
: 
o
training/SGD/mul_4MulSGD/momentum/readtraining/SGD/sub*
T0*&
_output_shapes
: 
q
training/SGD/add_1Addinput/kernel/readtraining/SGD/mul_4*&
_output_shapes
: *
T0
Ђ
training/SGD/mul_5Multraining/SGD/mul_1Btraining/SGD/gradients/input/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
r
training/SGD/sub_1Subtraining/SGD/add_1training/SGD/mul_5*
T0*&
_output_shapes
: 
М
training/SGD/Assign_1Assigninput/kerneltraining/SGD/sub_1*
use_locking(*
T0*
_class
loc:@input/kernel*
validate_shape(*&
_output_shapes
: 
o
training/SGD/mul_6MulSGD/momentum/readtraining/SGD/Variable_1/read*
_output_shapes
: *
T0

training/SGD/mul_7Multraining/SGD/mul_15training/SGD/gradients/input/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
f
training/SGD/sub_2Subtraining/SGD/mul_6training/SGD/mul_7*
T0*
_output_shapes
: 
Ц
training/SGD/Assign_2Assigntraining/SGD/Variable_1training/SGD/sub_2*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_1*
validate_shape(*
_output_shapes
: 
e
training/SGD/mul_8MulSGD/momentum/readtraining/SGD/sub_2*
T0*
_output_shapes
: 
c
training/SGD/add_2Addinput/bias/readtraining/SGD/mul_8*
T0*
_output_shapes
: 

training/SGD/mul_9Multraining/SGD/mul_15training/SGD/gradients/input/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
f
training/SGD/sub_3Subtraining/SGD/add_2training/SGD/mul_9*
T0*
_output_shapes
: 
Ќ
training/SGD/Assign_3Assign
input/biastraining/SGD/sub_3*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@input/bias
|
training/SGD/mul_10MulSGD/momentum/readtraining/SGD/Variable_2/read*&
_output_shapes
: @*
T0
І
training/SGD/mul_11Multraining/SGD/mul_1Etraining/SGD/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
t
training/SGD/sub_4Subtraining/SGD/mul_10training/SGD/mul_11*
T0*&
_output_shapes
: @
в
training/SGD/Assign_4Assigntraining/SGD/Variable_2training/SGD/sub_4*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_2*
validate_shape(*&
_output_shapes
: @
r
training/SGD/mul_12MulSGD/momentum/readtraining/SGD/sub_4*&
_output_shapes
: @*
T0
u
training/SGD/add_3Addconv2d_1/kernel/readtraining/SGD/mul_12*&
_output_shapes
: @*
T0
І
training/SGD/mul_13Multraining/SGD/mul_1Etraining/SGD/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
s
training/SGD/sub_5Subtraining/SGD/add_3training/SGD/mul_13*
T0*&
_output_shapes
: @
Т
training/SGD/Assign_5Assignconv2d_1/kerneltraining/SGD/sub_5*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
: @
p
training/SGD/mul_14MulSGD/momentum/readtraining/SGD/Variable_3/read*
T0*
_output_shapes
:@

training/SGD/mul_15Multraining/SGD/mul_18training/SGD/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
h
training/SGD/sub_6Subtraining/SGD/mul_14training/SGD/mul_15*
T0*
_output_shapes
:@
Ц
training/SGD/Assign_6Assigntraining/SGD/Variable_3training/SGD/sub_6**
_class 
loc:@training/SGD/Variable_3*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
f
training/SGD/mul_16MulSGD/momentum/readtraining/SGD/sub_6*
T0*
_output_shapes
:@
g
training/SGD/add_4Addconv2d_1/bias/readtraining/SGD/mul_16*
T0*
_output_shapes
:@

training/SGD/mul_17Multraining/SGD/mul_18training/SGD/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
g
training/SGD/sub_7Subtraining/SGD/add_4training/SGD/mul_17*
T0*
_output_shapes
:@
В
training/SGD/Assign_7Assignconv2d_1/biastraining/SGD/sub_7*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:@
|
training/SGD/mul_18MulSGD/momentum/readtraining/SGD/Variable_4/read*
T0*&
_output_shapes
:@
Є
training/SGD/mul_19Multraining/SGD/mul_1Ctraining/SGD/gradients/output/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
t
training/SGD/sub_8Subtraining/SGD/mul_18training/SGD/mul_19*
T0*&
_output_shapes
:@
в
training/SGD/Assign_8Assigntraining/SGD/Variable_4training/SGD/sub_8*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_4*
validate_shape(*&
_output_shapes
:@
r
training/SGD/mul_20MulSGD/momentum/readtraining/SGD/sub_8*
T0*&
_output_shapes
:@
s
training/SGD/add_5Addoutput/kernel/readtraining/SGD/mul_20*&
_output_shapes
:@*
T0
Є
training/SGD/mul_21Multraining/SGD/mul_1Ctraining/SGD/gradients/output/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
s
training/SGD/sub_9Subtraining/SGD/add_5training/SGD/mul_21*&
_output_shapes
:@*
T0
О
training/SGD/Assign_9Assignoutput/kerneltraining/SGD/sub_9*
use_locking(*
T0* 
_class
loc:@output/kernel*
validate_shape(*&
_output_shapes
:@
p
training/SGD/mul_22MulSGD/momentum/readtraining/SGD/Variable_5/read*
T0*
_output_shapes
:

training/SGD/mul_23Multraining/SGD/mul_16training/SGD/gradients/output/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
i
training/SGD/sub_10Subtraining/SGD/mul_22training/SGD/mul_23*
T0*
_output_shapes
:
Ш
training/SGD/Assign_10Assigntraining/SGD/Variable_5training/SGD/sub_10**
_class 
loc:@training/SGD/Variable_5*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
g
training/SGD/mul_24MulSGD/momentum/readtraining/SGD/sub_10*
T0*
_output_shapes
:
e
training/SGD/add_6Addoutput/bias/readtraining/SGD/mul_24*
_output_shapes
:*
T0

training/SGD/mul_25Multraining/SGD/mul_16training/SGD/gradients/output/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
h
training/SGD/sub_11Subtraining/SGD/add_6training/SGD/mul_25*
_output_shapes
:*
T0
А
training/SGD/Assign_11Assignoutput/biastraining/SGD/sub_11*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@output/bias*
validate_shape(
п
training/group_depsNoOp	^loss/mul^training/SGD/Assign^training/SGD/AssignAdd^training/SGD/Assign_1^training/SGD/Assign_10^training/SGD/Assign_11^training/SGD/Assign_2^training/SGD/Assign_3^training/SGD/Assign_4^training/SGD/Assign_5^training/SGD/Assign_6^training/SGD/Assign_7^training/SGD/Assign_8^training/SGD/Assign_9

IsVariableInitializedIsVariableInitializedinput/kernel*
_class
loc:@input/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitialized
input/bias*
dtype0*
_output_shapes
: *
_class
loc:@input/bias

IsVariableInitialized_2IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_3IsVariableInitializedconv2d_1/bias*
dtype0*
_output_shapes
: * 
_class
loc:@conv2d_1/bias

IsVariableInitialized_4IsVariableInitializedoutput/kernel*
_output_shapes
: * 
_class
loc:@output/kernel*
dtype0

IsVariableInitialized_5IsVariableInitializedoutput/bias*
_class
loc:@output/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_6IsVariableInitializedSGD/iterations*!
_class
loc:@SGD/iterations*
dtype0	*
_output_shapes
: 
x
IsVariableInitialized_7IsVariableInitializedSGD/lr*
_class
loc:@SGD/lr*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializedSGD/momentum*
_output_shapes
: *
_class
loc:@SGD/momentum*
dtype0
~
IsVariableInitialized_9IsVariableInitialized	SGD/decay*
dtype0*
_output_shapes
: *
_class
loc:@SGD/decay

IsVariableInitialized_10IsVariableInitializedtraining/SGD/Variable*
dtype0*
_output_shapes
: *(
_class
loc:@training/SGD/Variable

IsVariableInitialized_11IsVariableInitializedtraining/SGD/Variable_1**
_class 
loc:@training/SGD/Variable_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_12IsVariableInitializedtraining/SGD/Variable_2**
_class 
loc:@training/SGD/Variable_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_13IsVariableInitializedtraining/SGD/Variable_3*
dtype0*
_output_shapes
: **
_class 
loc:@training/SGD/Variable_3

IsVariableInitialized_14IsVariableInitializedtraining/SGD/Variable_4**
_class 
loc:@training/SGD/Variable_4*
dtype0*
_output_shapes
: 

IsVariableInitialized_15IsVariableInitializedtraining/SGD/Variable_5**
_class 
loc:@training/SGD/Variable_5*
dtype0*
_output_shapes
: 
Ї
initNoOp^SGD/decay/Assign^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^input/bias/Assign^input/kernel/Assign^output/bias/Assign^output/kernel/Assign^training/SGD/Variable/Assign^training/SGD/Variable_1/Assign^training/SGD/Variable_2/Assign^training/SGD/Variable_3/Assign^training/SGD/Variable_4/Assign^training/SGD/Variable_5/Assign"mу     {7	ЗвЙXі(зAJж
)ъ(
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
ь
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype
is_initialized
"
dtypetype
\
	LeakyRelu
features"T
activations"T"
alphafloat%ЭЬL>"
Ttype0:
2
n
LeakyReluGrad
	gradients"T
features"T
	backprops"T"
alphafloat%ЭЬL>"
Ttype0:
2
д
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
ю
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	

C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
x
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2		"
align_cornersbool( 
p
ResizeNearestNeighborGrad

grads"T
size
output"T"
Ttype:

2"
align_cornersbool( 
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
/
Sign
x"T
y"T"
Ttype:

2	
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *1.13.12v1.13.1-0-g6612da8951шО

input_inputPlaceholder*&
shape:џџџџџџџџџИа*
dtype0*1
_output_shapes
:џџџџџџџџџИа
s
input/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"             
]
input/random_uniform/minConst*
valueB
 *OSО*
dtype0*
_output_shapes
: 
]
input/random_uniform/maxConst*
valueB
 *OS>*
dtype0*
_output_shapes
: 
Ќ
"input/random_uniform/RandomUniformRandomUniforminput/random_uniform/shape*
dtype0*
seed2пу*&
_output_shapes
: *
seedБџх)*
T0
t
input/random_uniform/subSubinput/random_uniform/maxinput/random_uniform/min*
T0*
_output_shapes
: 

input/random_uniform/mulMul"input/random_uniform/RandomUniforminput/random_uniform/sub*
T0*&
_output_shapes
: 

input/random_uniformAddinput/random_uniform/mulinput/random_uniform/min*
T0*&
_output_shapes
: 

input/kernel
VariableV2*
dtype0*
	container *&
_output_shapes
: *
shape: *
shared_name 
М
input/kernel/AssignAssigninput/kernelinput/random_uniform*
_class
loc:@input/kernel*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0
}
input/kernel/readIdentityinput/kernel*
T0*
_class
loc:@input/kernel*&
_output_shapes
: 
X
input/ConstConst*
valueB *    *
dtype0*
_output_shapes
: 
v

input/bias
VariableV2*
	container *
_output_shapes
: *
shape: *
shared_name *
dtype0
Ё
input/bias/AssignAssign
input/biasinput/Const*
_class
loc:@input/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
k
input/bias/readIdentity
input/bias*
T0*
_class
loc:@input/bias*
_output_shapes
: 
p
input/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
х
input/convolutionConv2Dinput_inputinput/kernel/read*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа *
	dilations
*
T0*
strides
*
data_formatNHWC

input/BiasAddBiasAddinput/convolutioninput/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа 

leaky_re_lu_1/LeakyRelu	LeakyReluinput/BiasAdd*
T0*
alpha%>*1
_output_shapes
:џџџџџџџџџИа 
f
$dropout_1/keras_learning_phase/inputConst*
value	B
 Z *
dtype0
*
_output_shapes
: 

dropout_1/keras_learning_phasePlaceholderWithDefault$dropout_1/keras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 

dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes
: : *
T0

]
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
_output_shapes
: *
T0

c
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
: 
s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 

dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*1
_output_shapes
:џџџџџџџџџИа *
T0
й
dropout_1/cond/mul/SwitchSwitchleaky_re_lu_1/LeakyReludropout_1/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа 
z
dropout_1/cond/dropout/rateConst^dropout_1/cond/switch_t*
valueB
 *  >*
dtype0*
_output_shapes
: 
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:
{
dropout_1/cond/dropout/sub/xConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
}
dropout_1/cond/dropout/subSubdropout_1/cond/dropout/sub/xdropout_1/cond/dropout/rate*
T0*
_output_shapes
: 

)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ъ
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
T0*
dtype0*
seed2н*1
_output_shapes
:џџџџџџџџџИа *
seedБџх)
Ї
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ь
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*1
_output_shapes
:џџџџџџџџџИа *
T0
О
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*1
_output_shapes
:џџџџџџџџџИа 
 
dropout_1/cond/dropout/addAdddropout_1/cond/dropout/sub%dropout_1/cond/dropout/random_uniform*
T0*1
_output_shapes
:џџџџџџџџџИа 
}
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*
T0*1
_output_shapes
:џџџџџџџџџИа 

dropout_1/cond/dropout/truedivRealDivdropout_1/cond/muldropout_1/cond/dropout/sub*
T0*1
_output_shapes
:џџџџџџџџџИа 

dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/truedivdropout_1/cond/dropout/Floor*1
_output_shapes
:џџџџџџџџџИа *
T0
з
dropout_1/cond/Switch_1Switchleaky_re_lu_1/LeakyReludropout_1/cond/pred_id*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа *
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu

dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
T0*
N*3
_output_shapes!
:џџџџџџџџџИа : 
Ц
max_pooling2d_1/MaxPoolMaxPooldropout_1/cond/Merge*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ *
T0
v
conv2d_1/random_uniform/shapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
`
conv2d_1/random_uniform/minConst*
valueB
 *ЋЊЊН*
dtype0*
_output_shapes
: 
`
conv2d_1/random_uniform/maxConst*
valueB
 *ЋЊЊ=*
dtype0*
_output_shapes
: 
В
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
T0*
dtype0*
seed2їЄѓ*&
_output_shapes
: @*
seedБџх)
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
T0*
_output_shapes
: 

conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*
T0*&
_output_shapes
: @

conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*
T0*&
_output_shapes
: @

conv2d_1/kernel
VariableV2*
shape: @*
shared_name *
dtype0*
	container *&
_output_shapes
: @
Ш
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel

conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
T0
[
conv2d_1/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_1/bias
VariableV2*
	container *
_output_shapes
:@*
shape:@*
shared_name *
dtype0
­
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:@
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
s
"conv2d_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ї
conv2d_1/convolutionConv2Dmax_pooling2d_1/MaxPoolconv2d_1/kernel/read*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџЈ@

leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_1/BiasAdd*
T0*
alpha%>*1
_output_shapes
:џџџџџџџџџЈ@

dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
_output_shapes
: *
T0

[
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
_output_shapes
: *
T0

c
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
: 
s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*1
_output_shapes
:џџџџџџџџџЈ@*
T0
й
dropout_2/cond/mul/SwitchSwitchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@
z
dropout_2/cond/dropout/rateConst^dropout_2/cond/switch_t*
_output_shapes
: *
valueB
 *  >*
dtype0
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
T0*
out_type0*
_output_shapes
:
{
dropout_2/cond/dropout/sub/xConst^dropout_2/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
}
dropout_2/cond/dropout/subSubdropout_2/cond/dropout/sub/xdropout_2/cond/dropout/rate*
T0*
_output_shapes
: 

)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0

)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ъ
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
T0*
dtype0*
seed2из*1
_output_shapes
:џџџџџџџџџЈ@*
seedБџх)
Ї
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ь
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*
T0*1
_output_shapes
:џџџџџџџџџЈ@
О
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*
T0*1
_output_shapes
:џџџџџџџџџЈ@
 
dropout_2/cond/dropout/addAdddropout_2/cond/dropout/sub%dropout_2/cond/dropout/random_uniform*
T0*1
_output_shapes
:џџџџџџџџџЈ@
}
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*
T0*1
_output_shapes
:џџџџџџџџџЈ@

dropout_2/cond/dropout/truedivRealDivdropout_2/cond/muldropout_2/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџЈ@*
T0

dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/truedivdropout_2/cond/dropout/Floor*
T0*1
_output_shapes
:џџџџџџџџџЈ@
з
dropout_2/cond/Switch_1Switchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu

dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
T0*
N*3
_output_shapes!
:џџџџџџџџџЈ@: 
i
up_sampling2d_1/ShapeShapedropout_2/cond/Merge*
T0*
out_type0*
_output_shapes
:
m
#up_sampling2d_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Э
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape#up_sampling2d_1/strided_slice/stack%up_sampling2d_1/strided_slice/stack_1%up_sampling2d_1/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0
f
up_sampling2d_1/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
u
up_sampling2d_1/mulMulup_sampling2d_1/strided_sliceup_sampling2d_1/Const*
_output_shapes
:*
T0
К
%up_sampling2d_1/ResizeNearestNeighborResizeNearestNeighbordropout_2/cond/Mergeup_sampling2d_1/mul*
align_corners( *
T0*1
_output_shapes
:џџџџџџџџџИа@
t
output/random_uniform/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
^
output/random_uniform/minConst*
valueB
 *8JЬН*
dtype0*
_output_shapes
: 
^
output/random_uniform/maxConst*
valueB
 *8JЬ=*
dtype0*
_output_shapes
: 
Ў
#output/random_uniform/RandomUniformRandomUniformoutput/random_uniform/shape*
seedБџх)*
T0*
dtype0*
seed2юЙ*&
_output_shapes
:@
w
output/random_uniform/subSuboutput/random_uniform/maxoutput/random_uniform/min*
_output_shapes
: *
T0

output/random_uniform/mulMul#output/random_uniform/RandomUniformoutput/random_uniform/sub*
T0*&
_output_shapes
:@

output/random_uniformAddoutput/random_uniform/muloutput/random_uniform/min*
T0*&
_output_shapes
:@

output/kernel
VariableV2*
shared_name *
dtype0*
	container *&
_output_shapes
:@*
shape:@
Р
output/kernel/AssignAssignoutput/kerneloutput/random_uniform*
use_locking(*
T0* 
_class
loc:@output/kernel*
validate_shape(*&
_output_shapes
:@

output/kernel/readIdentityoutput/kernel*&
_output_shapes
:@*
T0* 
_class
loc:@output/kernel
Y
output/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
w
output/bias
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
Ѕ
output/bias/AssignAssignoutput/biasoutput/Const*
_class
loc:@output/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
n
output/bias/readIdentityoutput/bias*
T0*
_class
loc:@output/bias*
_output_shapes
:
q
 output/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      

output/convolutionConv2D%up_sampling2d_1/ResizeNearestNeighboroutput/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа

output/BiasAddBiasAddoutput/convolutionoutput/bias/read*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа*
T0
^
SGD/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
r
SGD/iterations
VariableV2*
	container *
_output_shapes
: *
shape: *
shared_name *
dtype0	
К
SGD/iterations/AssignAssignSGD/iterationsSGD/iterations/initial_value*
use_locking(*
T0	*!
_class
loc:@SGD/iterations*
validate_shape(*
_output_shapes
: 
s
SGD/iterations/readIdentitySGD/iterations*
_output_shapes
: *
T0	*!
_class
loc:@SGD/iterations
Y
SGD/lr/initial_valueConst*
valueB
 *Зб8*
dtype0*
_output_shapes
: 
j
SGD/lr
VariableV2*
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 

SGD/lr/AssignAssignSGD/lrSGD/lr/initial_value*
_class
loc:@SGD/lr*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
[
SGD/lr/readIdentitySGD/lr*
T0*
_class
loc:@SGD/lr*
_output_shapes
: 
_
SGD/momentum/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
p
SGD/momentum
VariableV2*
	container *
_output_shapes
: *
shape: *
shared_name *
dtype0
В
SGD/momentum/AssignAssignSGD/momentumSGD/momentum/initial_value*
T0*
_class
loc:@SGD/momentum*
validate_shape(*
_output_shapes
: *
use_locking(
m
SGD/momentum/readIdentitySGD/momentum*
_output_shapes
: *
T0*
_class
loc:@SGD/momentum
\
SGD/decay/initial_valueConst*
_output_shapes
: *
valueB
 *ЌХ'7*
dtype0
m
	SGD/decay
VariableV2*
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 
І
SGD/decay/AssignAssign	SGD/decaySGD/decay/initial_value*
use_locking(*
T0*
_class
loc:@SGD/decay*
validate_shape(*
_output_shapes
: 
d
SGD/decay/readIdentity	SGD/decay*
T0*
_class
loc:@SGD/decay*
_output_shapes
: 
Ж
output_targetPlaceholder*
dtype0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*?
shape6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
output_sample_weightsPlaceholder*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
v
loss/output_loss/subSuboutput/BiasAddoutput_target*
T0*1
_output_shapes
:џџџџџџџџџИа
m
loss/output_loss/AbsAbsloss/output_loss/sub*1
_output_shapes
:џџџџџџџџџИа*
T0
r
'loss/output_loss/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Б
loss/output_loss/MeanMeanloss/output_loss/Abs'loss/output_loss/Mean/reduction_indices*-
_output_shapes
:џџџџџџџџџИа*

Tidx0*
	keep_dims( *
T0
z
)loss/output_loss/Mean_1/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
Ќ
loss/output_loss/Mean_1Meanloss/output_loss/Mean)loss/output_loss/Mean_1/reduction_indices*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( *
T0
y
loss/output_loss/mulMulloss/output_loss/Mean_1output_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
`
loss/output_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/output_loss/NotEqualNotEqualoutput_sample_weightsloss/output_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ

loss/output_loss/CastCastloss/output_loss/NotEqual*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:џџџџџџџџџ
`
loss/output_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/output_loss/Mean_2Meanloss/output_loss/Castloss/output_loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 

loss/output_loss/truedivRealDivloss/output_loss/mulloss/output_loss/Mean_2*
T0*#
_output_shapes
:џџџџџџџџџ
b
loss/output_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/output_loss/Mean_3Meanloss/output_loss/truedivloss/output_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
U
loss/mulMul
loss/mul/xloss/output_loss/Mean_3*
T0*
_output_shapes
: 
|
training/SGD/gradients/ShapeConst*
_output_shapes
: *
_class
loc:@loss/mul*
valueB *
dtype0

 training/SGD/gradients/grad_ys_0Const*
_class
loc:@loss/mul*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Г
training/SGD/gradients/FillFilltraining/SGD/gradients/Shape training/SGD/gradients/grad_ys_0*
_class
loc:@loss/mul*

index_type0*
_output_shapes
: *
T0
Ѓ
(training/SGD/gradients/loss/mul_grad/MulMultraining/SGD/gradients/Fillloss/output_loss/Mean_3*
T0*
_class
loc:@loss/mul*
_output_shapes
: 

*training/SGD/gradients/loss/mul_grad/Mul_1Multraining/SGD/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
З
Atraining/SGD/gradients/loss/output_loss/Mean_3_grad/Reshape/shapeConst**
_class 
loc:@loss/output_loss/Mean_3*
valueB:*
dtype0*
_output_shapes
:

;training/SGD/gradients/loss/output_loss/Mean_3_grad/ReshapeReshape*training/SGD/gradients/loss/mul_grad/Mul_1Atraining/SGD/gradients/loss/output_loss/Mean_3_grad/Reshape/shape*
T0**
_class 
loc:@loss/output_loss/Mean_3*
Tshape0*
_output_shapes
:
Н
9training/SGD/gradients/loss/output_loss/Mean_3_grad/ShapeShapeloss/output_loss/truediv**
_class 
loc:@loss/output_loss/Mean_3*
out_type0*
_output_shapes
:*
T0
Є
8training/SGD/gradients/loss/output_loss/Mean_3_grad/TileTile;training/SGD/gradients/loss/output_loss/Mean_3_grad/Reshape9training/SGD/gradients/loss/output_loss/Mean_3_grad/Shape*

Tmultiples0*
T0**
_class 
loc:@loss/output_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
П
;training/SGD/gradients/loss/output_loss/Mean_3_grad/Shape_1Shapeloss/output_loss/truediv**
_class 
loc:@loss/output_loss/Mean_3*
out_type0*
_output_shapes
:*
T0
Њ
;training/SGD/gradients/loss/output_loss/Mean_3_grad/Shape_2Const**
_class 
loc:@loss/output_loss/Mean_3*
valueB *
dtype0*
_output_shapes
: 
Џ
9training/SGD/gradients/loss/output_loss/Mean_3_grad/ConstConst*
_output_shapes
:**
_class 
loc:@loss/output_loss/Mean_3*
valueB: *
dtype0
Ђ
8training/SGD/gradients/loss/output_loss/Mean_3_grad/ProdProd;training/SGD/gradients/loss/output_loss/Mean_3_grad/Shape_19training/SGD/gradients/loss/output_loss/Mean_3_grad/Const*
	keep_dims( *

Tidx0*
T0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
: 
Б
;training/SGD/gradients/loss/output_loss/Mean_3_grad/Const_1Const**
_class 
loc:@loss/output_loss/Mean_3*
valueB: *
dtype0*
_output_shapes
:
І
:training/SGD/gradients/loss/output_loss/Mean_3_grad/Prod_1Prod;training/SGD/gradients/loss/output_loss/Mean_3_grad/Shape_2;training/SGD/gradients/loss/output_loss/Mean_3_grad/Const_1*
T0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
: *
	keep_dims( *

Tidx0
Ћ
=training/SGD/gradients/loss/output_loss/Mean_3_grad/Maximum/yConst**
_class 
loc:@loss/output_loss/Mean_3*
value	B :*
dtype0*
_output_shapes
: 

;training/SGD/gradients/loss/output_loss/Mean_3_grad/MaximumMaximum:training/SGD/gradients/loss/output_loss/Mean_3_grad/Prod_1=training/SGD/gradients/loss/output_loss/Mean_3_grad/Maximum/y*
T0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
: 

<training/SGD/gradients/loss/output_loss/Mean_3_grad/floordivFloorDiv8training/SGD/gradients/loss/output_loss/Mean_3_grad/Prod;training/SGD/gradients/loss/output_loss/Mean_3_grad/Maximum*
T0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
: 
ъ
8training/SGD/gradients/loss/output_loss/Mean_3_grad/CastCast<training/SGD/gradients/loss/output_loss/Mean_3_grad/floordiv*

SrcT0**
_class 
loc:@loss/output_loss/Mean_3*
Truncate( *

DstT0*
_output_shapes
: 

;training/SGD/gradients/loss/output_loss/Mean_3_grad/truedivRealDiv8training/SGD/gradients/loss/output_loss/Mean_3_grad/Tile8training/SGD/gradients/loss/output_loss/Mean_3_grad/Cast*
T0**
_class 
loc:@loss/output_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Л
:training/SGD/gradients/loss/output_loss/truediv_grad/ShapeShapeloss/output_loss/mul*
T0*+
_class!
loc:@loss/output_loss/truediv*
out_type0*
_output_shapes
:
Ќ
<training/SGD/gradients/loss/output_loss/truediv_grad/Shape_1Const*+
_class!
loc:@loss/output_loss/truediv*
valueB *
dtype0*
_output_shapes
: 
Ч
Jtraining/SGD/gradients/loss/output_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs:training/SGD/gradients/loss/output_loss/truediv_grad/Shape<training/SGD/gradients/loss/output_loss/truediv_grad/Shape_1*
T0*+
_class!
loc:@loss/output_loss/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ј
<training/SGD/gradients/loss/output_loss/truediv_grad/RealDivRealDiv;training/SGD/gradients/loss/output_loss/Mean_3_grad/truedivloss/output_loss/Mean_2*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:џџџџџџџџџ*
T0
Ж
8training/SGD/gradients/loss/output_loss/truediv_grad/SumSum<training/SGD/gradients/loss/output_loss/truediv_grad/RealDivJtraining/SGD/gradients/loss/output_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/output_loss/truediv*
_output_shapes
:
І
<training/SGD/gradients/loss/output_loss/truediv_grad/ReshapeReshape8training/SGD/gradients/loss/output_loss/truediv_grad/Sum:training/SGD/gradients/loss/output_loss/truediv_grad/Shape*
T0*+
_class!
loc:@loss/output_loss/truediv*
Tshape0*#
_output_shapes
:џџџџџџџџџ
А
8training/SGD/gradients/loss/output_loss/truediv_grad/NegNegloss/output_loss/mul*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:џџџџџџџџџ*
T0
ї
>training/SGD/gradients/loss/output_loss/truediv_grad/RealDiv_1RealDiv8training/SGD/gradients/loss/output_loss/truediv_grad/Negloss/output_loss/Mean_2*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:џџџџџџџџџ
§
>training/SGD/gradients/loss/output_loss/truediv_grad/RealDiv_2RealDiv>training/SGD/gradients/loss/output_loss/truediv_grad/RealDiv_1loss/output_loss/Mean_2*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:џџџџџџџџџ*
T0

8training/SGD/gradients/loss/output_loss/truediv_grad/mulMul;training/SGD/gradients/loss/output_loss/Mean_3_grad/truediv>training/SGD/gradients/loss/output_loss/truediv_grad/RealDiv_2*#
_output_shapes
:џџџџџџџџџ*
T0*+
_class!
loc:@loss/output_loss/truediv
Ж
:training/SGD/gradients/loss/output_loss/truediv_grad/Sum_1Sum8training/SGD/gradients/loss/output_loss/truediv_grad/mulLtraining/SGD/gradients/loss/output_loss/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/output_loss/truediv

>training/SGD/gradients/loss/output_loss/truediv_grad/Reshape_1Reshape:training/SGD/gradients/loss/output_loss/truediv_grad/Sum_1<training/SGD/gradients/loss/output_loss/truediv_grad/Shape_1*
T0*+
_class!
loc:@loss/output_loss/truediv*
Tshape0*
_output_shapes
: 
Ж
6training/SGD/gradients/loss/output_loss/mul_grad/ShapeShapeloss/output_loss/Mean_1*
_output_shapes
:*
T0*'
_class
loc:@loss/output_loss/mul*
out_type0
Ж
8training/SGD/gradients/loss/output_loss/mul_grad/Shape_1Shapeoutput_sample_weights*
T0*'
_class
loc:@loss/output_loss/mul*
out_type0*
_output_shapes
:
З
Ftraining/SGD/gradients/loss/output_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6training/SGD/gradients/loss/output_loss/mul_grad/Shape8training/SGD/gradients/loss/output_loss/mul_grad/Shape_1*
T0*'
_class
loc:@loss/output_loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ч
4training/SGD/gradients/loss/output_loss/mul_grad/MulMul<training/SGD/gradients/loss/output_loss/truediv_grad/Reshapeoutput_sample_weights*
T0*'
_class
loc:@loss/output_loss/mul*#
_output_shapes
:џџџџџџџџџ
Ђ
4training/SGD/gradients/loss/output_loss/mul_grad/SumSum4training/SGD/gradients/loss/output_loss/mul_grad/MulFtraining/SGD/gradients/loss/output_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*'
_class
loc:@loss/output_loss/mul*
_output_shapes
:

8training/SGD/gradients/loss/output_loss/mul_grad/ReshapeReshape4training/SGD/gradients/loss/output_loss/mul_grad/Sum6training/SGD/gradients/loss/output_loss/mul_grad/Shape*'
_class
loc:@loss/output_loss/mul*
Tshape0*#
_output_shapes
:џџџџџџџџџ*
T0
ы
6training/SGD/gradients/loss/output_loss/mul_grad/Mul_1Mulloss/output_loss/Mean_1<training/SGD/gradients/loss/output_loss/truediv_grad/Reshape*
T0*'
_class
loc:@loss/output_loss/mul*#
_output_shapes
:џџџџџџџџџ
Ј
6training/SGD/gradients/loss/output_loss/mul_grad/Sum_1Sum6training/SGD/gradients/loss/output_loss/mul_grad/Mul_1Htraining/SGD/gradients/loss/output_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*'
_class
loc:@loss/output_loss/mul*
_output_shapes
:

:training/SGD/gradients/loss/output_loss/mul_grad/Reshape_1Reshape6training/SGD/gradients/loss/output_loss/mul_grad/Sum_18training/SGD/gradients/loss/output_loss/mul_grad/Shape_1*'
_class
loc:@loss/output_loss/mul*
Tshape0*#
_output_shapes
:џџџџџџџџџ*
T0
К
9training/SGD/gradients/loss/output_loss/Mean_1_grad/ShapeShapeloss/output_loss/Mean*
T0**
_class 
loc:@loss/output_loss/Mean_1*
out_type0*
_output_shapes
:
І
8training/SGD/gradients/loss/output_loss/Mean_1_grad/SizeConst**
_class 
loc:@loss/output_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
є
7training/SGD/gradients/loss/output_loss/Mean_1_grad/addAdd)loss/output_loss/Mean_1/reduction_indices8training/SGD/gradients/loss/output_loss/Mean_1_grad/Size*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:

7training/SGD/gradients/loss/output_loss/Mean_1_grad/modFloorMod7training/SGD/gradients/loss/output_loss/Mean_1_grad/add8training/SGD/gradients/loss/output_loss/Mean_1_grad/Size*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:
Б
;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_1Const**
_class 
loc:@loss/output_loss/Mean_1*
valueB:*
dtype0*
_output_shapes
:
­
?training/SGD/gradients/loss/output_loss/Mean_1_grad/range/startConst*
dtype0*
_output_shapes
: **
_class 
loc:@loss/output_loss/Mean_1*
value	B : 
­
?training/SGD/gradients/loss/output_loss/Mean_1_grad/range/deltaConst**
_class 
loc:@loss/output_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
в
9training/SGD/gradients/loss/output_loss/Mean_1_grad/rangeRange?training/SGD/gradients/loss/output_loss/Mean_1_grad/range/start8training/SGD/gradients/loss/output_loss/Mean_1_grad/Size?training/SGD/gradients/loss/output_loss/Mean_1_grad/range/delta**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:*

Tidx0
Ќ
>training/SGD/gradients/loss/output_loss/Mean_1_grad/Fill/valueConst*
_output_shapes
: **
_class 
loc:@loss/output_loss/Mean_1*
value	B :*
dtype0
 
8training/SGD/gradients/loss/output_loss/Mean_1_grad/FillFill;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_1>training/SGD/gradients/loss/output_loss/Mean_1_grad/Fill/value*
_output_shapes
:*
T0**
_class 
loc:@loss/output_loss/Mean_1*

index_type0

Atraining/SGD/gradients/loss/output_loss/Mean_1_grad/DynamicStitchDynamicStitch9training/SGD/gradients/loss/output_loss/Mean_1_grad/range7training/SGD/gradients/loss/output_loss/Mean_1_grad/mod9training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape8training/SGD/gradients/loss/output_loss/Mean_1_grad/Fill**
_class 
loc:@loss/output_loss/Mean_1*
N*
_output_shapes
:*
T0
Ћ
=training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum/yConst**
_class 
loc:@loss/output_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 

;training/SGD/gradients/loss/output_loss/Mean_1_grad/MaximumMaximumAtraining/SGD/gradients/loss/output_loss/Mean_1_grad/DynamicStitch=training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum/y*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:

<training/SGD/gradients/loss/output_loss/Mean_1_grad/floordivFloorDiv9training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape;training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum*
_output_shapes
:*
T0**
_class 
loc:@loss/output_loss/Mean_1
Х
;training/SGD/gradients/loss/output_loss/Mean_1_grad/ReshapeReshape8training/SGD/gradients/loss/output_loss/mul_grad/ReshapeAtraining/SGD/gradients/loss/output_loss/Mean_1_grad/DynamicStitch*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0**
_class 
loc:@loss/output_loss/Mean_1*
Tshape0
С
8training/SGD/gradients/loss/output_loss/Mean_1_grad/TileTile;training/SGD/gradients/loss/output_loss/Mean_1_grad/Reshape<training/SGD/gradients/loss/output_loss/Mean_1_grad/floordiv**
_class 
loc:@loss/output_loss/Mean_1*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*

Tmultiples0*
T0
М
;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_2Shapeloss/output_loss/Mean*
_output_shapes
:*
T0**
_class 
loc:@loss/output_loss/Mean_1*
out_type0
О
;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_3Shapeloss/output_loss/Mean_1*
T0**
_class 
loc:@loss/output_loss/Mean_1*
out_type0*
_output_shapes
:
Џ
9training/SGD/gradients/loss/output_loss/Mean_1_grad/ConstConst**
_class 
loc:@loss/output_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
Ђ
8training/SGD/gradients/loss/output_loss/Mean_1_grad/ProdProd;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_29training/SGD/gradients/loss/output_loss/Mean_1_grad/Const*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: *
	keep_dims( *

Tidx0
Б
;training/SGD/gradients/loss/output_loss/Mean_1_grad/Const_1Const**
_class 
loc:@loss/output_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
І
:training/SGD/gradients/loss/output_loss/Mean_1_grad/Prod_1Prod;training/SGD/gradients/loss/output_loss/Mean_1_grad/Shape_3;training/SGD/gradients/loss/output_loss/Mean_1_grad/Const_1*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: *
	keep_dims( *

Tidx0
­
?training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum_1/yConst*
_output_shapes
: **
_class 
loc:@loss/output_loss/Mean_1*
value	B :*
dtype0

=training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum_1Maximum:training/SGD/gradients/loss/output_loss/Mean_1_grad/Prod_1?training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum_1/y*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: 

>training/SGD/gradients/loss/output_loss/Mean_1_grad/floordiv_1FloorDiv8training/SGD/gradients/loss/output_loss/Mean_1_grad/Prod=training/SGD/gradients/loss/output_loss/Mean_1_grad/Maximum_1*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: 
ь
8training/SGD/gradients/loss/output_loss/Mean_1_grad/CastCast>training/SGD/gradients/loss/output_loss/Mean_1_grad/floordiv_1*

SrcT0**
_class 
loc:@loss/output_loss/Mean_1*
Truncate( *

DstT0*
_output_shapes
: 

;training/SGD/gradients/loss/output_loss/Mean_1_grad/truedivRealDiv8training/SGD/gradients/loss/output_loss/Mean_1_grad/Tile8training/SGD/gradients/loss/output_loss/Mean_1_grad/Cast*
T0**
_class 
loc:@loss/output_loss/Mean_1*-
_output_shapes
:џџџџџџџџџИа
Е
7training/SGD/gradients/loss/output_loss/Mean_grad/ShapeShapeloss/output_loss/Abs*
_output_shapes
:*
T0*(
_class
loc:@loss/output_loss/Mean*
out_type0
Ђ
6training/SGD/gradients/loss/output_loss/Mean_grad/SizeConst*(
_class
loc:@loss/output_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
ш
5training/SGD/gradients/loss/output_loss/Mean_grad/addAdd'loss/output_loss/Mean/reduction_indices6training/SGD/gradients/loss/output_loss/Mean_grad/Size*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
ћ
5training/SGD/gradients/loss/output_loss/Mean_grad/modFloorMod5training/SGD/gradients/loss/output_loss/Mean_grad/add6training/SGD/gradients/loss/output_loss/Mean_grad/Size*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
І
9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_1Const*
dtype0*
_output_shapes
: *(
_class
loc:@loss/output_loss/Mean*
valueB 
Љ
=training/SGD/gradients/loss/output_loss/Mean_grad/range/startConst*(
_class
loc:@loss/output_loss/Mean*
value	B : *
dtype0*
_output_shapes
: 
Љ
=training/SGD/gradients/loss/output_loss/Mean_grad/range/deltaConst*(
_class
loc:@loss/output_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
Ш
7training/SGD/gradients/loss/output_loss/Mean_grad/rangeRange=training/SGD/gradients/loss/output_loss/Mean_grad/range/start6training/SGD/gradients/loss/output_loss/Mean_grad/Size=training/SGD/gradients/loss/output_loss/Mean_grad/range/delta*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:*

Tidx0
Ј
<training/SGD/gradients/loss/output_loss/Mean_grad/Fill/valueConst*(
_class
loc:@loss/output_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

6training/SGD/gradients/loss/output_loss/Mean_grad/FillFill9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_1<training/SGD/gradients/loss/output_loss/Mean_grad/Fill/value*
T0*(
_class
loc:@loss/output_loss/Mean*

index_type0*
_output_shapes
: 

?training/SGD/gradients/loss/output_loss/Mean_grad/DynamicStitchDynamicStitch7training/SGD/gradients/loss/output_loss/Mean_grad/range5training/SGD/gradients/loss/output_loss/Mean_grad/mod7training/SGD/gradients/loss/output_loss/Mean_grad/Shape6training/SGD/gradients/loss/output_loss/Mean_grad/Fill*
T0*(
_class
loc:@loss/output_loss/Mean*
N*
_output_shapes
:
Ї
;training/SGD/gradients/loss/output_loss/Mean_grad/Maximum/yConst*(
_class
loc:@loss/output_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

9training/SGD/gradients/loss/output_loss/Mean_grad/MaximumMaximum?training/SGD/gradients/loss/output_loss/Mean_grad/DynamicStitch;training/SGD/gradients/loss/output_loss/Mean_grad/Maximum/y*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:*
T0

:training/SGD/gradients/loss/output_loss/Mean_grad/floordivFloorDiv7training/SGD/gradients/loss/output_loss/Mean_grad/Shape9training/SGD/gradients/loss/output_loss/Mean_grad/Maximum*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:
Я
9training/SGD/gradients/loss/output_loss/Mean_grad/ReshapeReshape;training/SGD/gradients/loss/output_loss/Mean_1_grad/truediv?training/SGD/gradients/loss/output_loss/Mean_grad/DynamicStitch*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*(
_class
loc:@loss/output_loss/Mean*
Tshape0
Ц
6training/SGD/gradients/loss/output_loss/Mean_grad/TileTile9training/SGD/gradients/loss/output_loss/Mean_grad/Reshape:training/SGD/gradients/loss/output_loss/Mean_grad/floordiv*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*

Tmultiples0*
T0*(
_class
loc:@loss/output_loss/Mean
З
9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_2Shapeloss/output_loss/Abs*
T0*(
_class
loc:@loss/output_loss/Mean*
out_type0*
_output_shapes
:
И
9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_3Shapeloss/output_loss/Mean*
T0*(
_class
loc:@loss/output_loss/Mean*
out_type0*
_output_shapes
:
Ћ
7training/SGD/gradients/loss/output_loss/Mean_grad/ConstConst*(
_class
loc:@loss/output_loss/Mean*
valueB: *
dtype0*
_output_shapes
:

6training/SGD/gradients/loss/output_loss/Mean_grad/ProdProd9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_27training/SGD/gradients/loss/output_loss/Mean_grad/Const*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: *
	keep_dims( *

Tidx0
­
9training/SGD/gradients/loss/output_loss/Mean_grad/Const_1Const*(
_class
loc:@loss/output_loss/Mean*
valueB: *
dtype0*
_output_shapes
:

8training/SGD/gradients/loss/output_loss/Mean_grad/Prod_1Prod9training/SGD/gradients/loss/output_loss/Mean_grad/Shape_39training/SGD/gradients/loss/output_loss/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
Љ
=training/SGD/gradients/loss/output_loss/Mean_grad/Maximum_1/yConst*(
_class
loc:@loss/output_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

;training/SGD/gradients/loss/output_loss/Mean_grad/Maximum_1Maximum8training/SGD/gradients/loss/output_loss/Mean_grad/Prod_1=training/SGD/gradients/loss/output_loss/Mean_grad/Maximum_1/y*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 

<training/SGD/gradients/loss/output_loss/Mean_grad/floordiv_1FloorDiv6training/SGD/gradients/loss/output_loss/Mean_grad/Prod;training/SGD/gradients/loss/output_loss/Mean_grad/Maximum_1*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
ц
6training/SGD/gradients/loss/output_loss/Mean_grad/CastCast<training/SGD/gradients/loss/output_loss/Mean_grad/floordiv_1*(
_class
loc:@loss/output_loss/Mean*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0

9training/SGD/gradients/loss/output_loss/Mean_grad/truedivRealDiv6training/SGD/gradients/loss/output_loss/Mean_grad/Tile6training/SGD/gradients/loss/output_loss/Mean_grad/Cast*
T0*(
_class
loc:@loss/output_loss/Mean*1
_output_shapes
:џџџџџџџџџИа
И
5training/SGD/gradients/loss/output_loss/Abs_grad/SignSignloss/output_loss/sub*
T0*'
_class
loc:@loss/output_loss/Abs*1
_output_shapes
:џџџџџџџџџИа

4training/SGD/gradients/loss/output_loss/Abs_grad/mulMul9training/SGD/gradients/loss/output_loss/Mean_grad/truediv5training/SGD/gradients/loss/output_loss/Abs_grad/Sign*
T0*'
_class
loc:@loss/output_loss/Abs*1
_output_shapes
:џџџџџџџџџИа
­
6training/SGD/gradients/loss/output_loss/sub_grad/ShapeShapeoutput/BiasAdd*
T0*'
_class
loc:@loss/output_loss/sub*
out_type0*
_output_shapes
:
Ў
8training/SGD/gradients/loss/output_loss/sub_grad/Shape_1Shapeoutput_target*'
_class
loc:@loss/output_loss/sub*
out_type0*
_output_shapes
:*
T0
З
Ftraining/SGD/gradients/loss/output_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs6training/SGD/gradients/loss/output_loss/sub_grad/Shape8training/SGD/gradients/loss/output_loss/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*'
_class
loc:@loss/output_loss/sub
Ђ
4training/SGD/gradients/loss/output_loss/sub_grad/SumSum4training/SGD/gradients/loss/output_loss/Abs_grad/mulFtraining/SGD/gradients/loss/output_loss/sub_grad/BroadcastGradientArgs*
T0*'
_class
loc:@loss/output_loss/sub*
_output_shapes
:*
	keep_dims( *

Tidx0
Є
8training/SGD/gradients/loss/output_loss/sub_grad/ReshapeReshape4training/SGD/gradients/loss/output_loss/sub_grad/Sum6training/SGD/gradients/loss/output_loss/sub_grad/Shape*1
_output_shapes
:џџџџџџџџџИа*
T0*'
_class
loc:@loss/output_loss/sub*
Tshape0
І
6training/SGD/gradients/loss/output_loss/sub_grad/Sum_1Sum4training/SGD/gradients/loss/output_loss/Abs_grad/mulHtraining/SGD/gradients/loss/output_loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*'
_class
loc:@loss/output_loss/sub
П
4training/SGD/gradients/loss/output_loss/sub_grad/NegNeg6training/SGD/gradients/loss/output_loss/sub_grad/Sum_1*
_output_shapes
:*
T0*'
_class
loc:@loss/output_loss/sub
С
:training/SGD/gradients/loss/output_loss/sub_grad/Reshape_1Reshape4training/SGD/gradients/loss/output_loss/sub_grad/Neg8training/SGD/gradients/loss/output_loss/sub_grad/Shape_1*'
_class
loc:@loss/output_loss/sub*
Tshape0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0
о
6training/SGD/gradients/output/BiasAdd_grad/BiasAddGradBiasAddGrad8training/SGD/gradients/loss/output_loss/sub_grad/Reshape*
T0*!
_class
loc:@output/BiasAdd*
data_formatNHWC*
_output_shapes
:
х
5training/SGD/gradients/output/convolution_grad/ShapeNShapeN%up_sampling2d_1/ResizeNearestNeighboroutput/kernel/read*
T0*%
_class
loc:@output/convolution*
out_type0*
N* 
_output_shapes
::
Џ
Btraining/SGD/gradients/output/convolution_grad/Conv2DBackpropInputConv2DBackpropInput5training/SGD/gradients/output/convolution_grad/ShapeNoutput/kernel/read8training/SGD/gradients/loss/output_loss/sub_grad/Reshape*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа@*
	dilations
*
T0*%
_class
loc:@output/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
Л
Ctraining/SGD/gradients/output/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_1/ResizeNearestNeighbor7training/SGD/gradients/output/convolution_grad/ShapeN:18training/SGD/gradients/loss/output_loss/sub_grad/Reshape*&
_output_shapes
:@*
	dilations
*
T0*%
_class
loc:@output/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ы
`training/SGD/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*
valueB"  (  *
dtype0*
_output_shapes
:
Љ
[training/SGD/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradBtraining/SGD/gradients/output/convolution_grad/Conv2DBackpropInput`training/SGD/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*1
_output_shapes
:џџџџџџџџџЈ@*
align_corners( *
T0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor
Ь
:training/SGD/gradients/dropout_2/cond/Merge_grad/cond_gradSwitch[training/SGD/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGraddropout_2/cond/pred_id*
T0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@
н
training/SGD/gradients/SwitchSwitchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@
Д
training/SGD/gradients/IdentityIdentitytraining/SGD/gradients/Switch:1**
_class 
loc:@leaky_re_lu_2/LeakyRelu*1
_output_shapes
:џџџџџџџџџЈ@*
T0
Љ
training/SGD/gradients/Shape_1Shapetraining/SGD/gradients/Switch:1*
_output_shapes
:*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
out_type0
Е
"training/SGD/gradients/zeros/ConstConst ^training/SGD/gradients/Identity*
dtype0*
_output_shapes
: **
_class 
loc:@leaky_re_lu_2/LeakyRelu*
valueB
 *    
т
training/SGD/gradients/zerosFilltraining/SGD/gradients/Shape_1"training/SGD/gradients/zeros/Const*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*

index_type0*1
_output_shapes
:џџџџџџџџџЈ@

=training/SGD/gradients/dropout_2/cond/Switch_1_grad/cond_gradMerge:training/SGD/gradients/dropout_2/cond/Merge_grad/cond_gradtraining/SGD/gradients/zeros*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџЈ@: 
Щ
<training/SGD/gradients/dropout_2/cond/dropout/mul_grad/ShapeShapedropout_2/cond/dropout/truediv*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
out_type0*
_output_shapes
:
Щ
>training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Shape_1Shapedropout_2/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
out_type0*
_output_shapes
:
Я
Ltraining/SGD/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Shape>training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

:training/SGD/gradients/dropout_2/cond/dropout/mul_grad/MulMul<training/SGD/gradients/dropout_2/cond/Merge_grad/cond_grad:1dropout_2/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџЈ@
К
:training/SGD/gradients/dropout_2/cond/dropout/mul_grad/SumSum:training/SGD/gradients/dropout_2/cond/dropout/mul_grad/MulLtraining/SGD/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:
М
>training/SGD/gradients/dropout_2/cond/dropout/mul_grad/ReshapeReshape:training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Sum<training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Shape*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
Tshape0*1
_output_shapes
:џџџџџџџџџЈ@

<training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Muldropout_2/cond/dropout/truediv<training/SGD/gradients/dropout_2/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџЈ@
Р
<training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Sum_1Sum<training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Ntraining/SGD/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Т
@training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape<training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Sum_1>training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
Tshape0*1
_output_shapes
:џџџџџџџџџЈ@
Х
@training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/ShapeShapedropout_2/cond/mul*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
out_type0*
_output_shapes
:
И
Btraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Shape_1Const*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
valueB *
dtype0*
_output_shapes
: 
п
Ptraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs@training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/ShapeBtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv

Btraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/RealDivRealDiv>training/SGD/gradients/dropout_2/cond/dropout/mul_grad/Reshapedropout_2/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@
Ю
>training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/SumSumBtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/RealDivPtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
_output_shapes
:
Ь
Btraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/ReshapeReshape>training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Sum@training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Shape*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
Tshape0*1
_output_shapes
:џџџџџџџџџЈ@
Ш
>training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/NegNegdropout_2/cond/mul*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@

Dtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_1RealDiv>training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Negdropout_2/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџЈ@*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv
 
Dtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_2RealDivDtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_1dropout_2/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@
К
>training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/mulMul>training/SGD/gradients/dropout_2/cond/dropout/mul_grad/ReshapeDtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_2*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@
Ю
@training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Sum_1Sum>training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/mulRtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv
З
Dtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Reshape_1Reshape@training/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Sum_1Btraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Shape_1*
_output_shapes
: *
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
Tshape0
Ж
4training/SGD/gradients/dropout_2/cond/mul_grad/ShapeShapedropout_2/cond/mul/Switch:1*
_output_shapes
:*
T0*%
_class
loc:@dropout_2/cond/mul*
out_type0
 
6training/SGD/gradients/dropout_2/cond/mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *%
_class
loc:@dropout_2/cond/mul*
valueB 
Џ
Dtraining/SGD/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4training/SGD/gradients/dropout_2/cond/mul_grad/Shape6training/SGD/gradients/dropout_2/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_2/cond/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
і
2training/SGD/gradients/dropout_2/cond/mul_grad/MulMulBtraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Reshapedropout_2/cond/mul/y*%
_class
loc:@dropout_2/cond/mul*1
_output_shapes
:џџџџџџџџџЈ@*
T0

2training/SGD/gradients/dropout_2/cond/mul_grad/SumSum2training/SGD/gradients/dropout_2/cond/mul_grad/MulDtraining/SGD/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:

6training/SGD/gradients/dropout_2/cond/mul_grad/ReshapeReshape2training/SGD/gradients/dropout_2/cond/mul_grad/Sum4training/SGD/gradients/dropout_2/cond/mul_grad/Shape*1
_output_shapes
:џџџџџџџџџЈ@*
T0*%
_class
loc:@dropout_2/cond/mul*
Tshape0
џ
4training/SGD/gradients/dropout_2/cond/mul_grad/Mul_1Muldropout_2/cond/mul/Switch:1Btraining/SGD/gradients/dropout_2/cond/dropout/truediv_grad/Reshape*
T0*%
_class
loc:@dropout_2/cond/mul*1
_output_shapes
:џџџџџџџџџЈ@
 
4training/SGD/gradients/dropout_2/cond/mul_grad/Sum_1Sum4training/SGD/gradients/dropout_2/cond/mul_grad/Mul_1Ftraining/SGD/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0

8training/SGD/gradients/dropout_2/cond/mul_grad/Reshape_1Reshape4training/SGD/gradients/dropout_2/cond/mul_grad/Sum_16training/SGD/gradients/dropout_2/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_2/cond/mul*
Tshape0*
_output_shapes
: 
п
training/SGD/gradients/Switch_1Switchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu
Ж
!training/SGD/gradients/Identity_1Identitytraining/SGD/gradients/Switch_1*1
_output_shapes
:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu
Љ
training/SGD/gradients/Shape_2Shapetraining/SGD/gradients/Switch_1*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
out_type0*
_output_shapes
:
Й
$training/SGD/gradients/zeros_1/ConstConst"^training/SGD/gradients/Identity_1**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
valueB
 *    *
dtype0*
_output_shapes
: 
ц
training/SGD/gradients/zeros_1Filltraining/SGD/gradients/Shape_2$training/SGD/gradients/zeros_1/Const*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*

index_type0*1
_output_shapes
:џџџџџџџџџЈ@

?training/SGD/gradients/dropout_2/cond/mul/Switch_grad/cond_gradMergetraining/SGD/gradients/zeros_16training/SGD/gradients/dropout_2/cond/mul_grad/Reshape*3
_output_shapes!
:џџџџџџџџџЈ@: *
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N

training/SGD/gradients/AddNAddN=training/SGD/gradients/dropout_2/cond/Switch_1_grad/cond_grad?training/SGD/gradients/dropout_2/cond/mul/Switch_grad/cond_grad*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N*1
_output_shapes
:џџџџџџџџџЈ@
љ
Atraining/SGD/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGradLeakyReluGradtraining/SGD/gradients/AddNconv2d_1/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
alpha%>*1
_output_shapes
:џџџџџџџџџЈ@
ы
8training/SGD/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGradAtraining/SGD/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_1/BiasAdd*
data_formatNHWC*
_output_shapes
:@
н
7training/SGD/gradients/conv2d_1/convolution_grad/ShapeNShapeNmax_pooling2d_1/MaxPoolconv2d_1/kernel/read*
N* 
_output_shapes
::*
T0*'
_class
loc:@conv2d_1/convolution*
out_type0
Р
Dtraining/SGD/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput7training/SGD/gradients/conv2d_1/convolution_grad/ShapeNconv2d_1/kernel/readAtraining/SGD/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ *
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
М
Etraining/SGD/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_1/MaxPool9training/SGD/gradients/conv2d_1/convolution_grad/ShapeN:1Atraining/SGD/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*&
_output_shapes
: @*
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
§
?training/SGD/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGraddropout_1/cond/Mergemax_pooling2d_1/MaxPoolDtraining/SGD/gradients/conv2d_1/convolution_grad/Conv2DBackpropInput*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа *
T0**
_class 
loc:@max_pooling2d_1/MaxPool*
data_formatNHWC*
strides

Ђ
:training/SGD/gradients/dropout_1/cond/Merge_grad/cond_gradSwitch?training/SGD/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGraddropout_1/cond/pred_id*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа *
T0**
_class 
loc:@max_pooling2d_1/MaxPool
п
training/SGD/gradients/Switch_2Switchleaky_re_lu_1/LeakyReludropout_1/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа 
И
!training/SGD/gradients/Identity_2Identity!training/SGD/gradients/Switch_2:1*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*1
_output_shapes
:џџџџџџџџџИа 
Ћ
training/SGD/gradients/Shape_3Shape!training/SGD/gradients/Switch_2:1*
_output_shapes
:*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
out_type0
Й
$training/SGD/gradients/zeros_2/ConstConst"^training/SGD/gradients/Identity_2**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
valueB
 *    *
dtype0*
_output_shapes
: 
ц
training/SGD/gradients/zeros_2Filltraining/SGD/gradients/Shape_3$training/SGD/gradients/zeros_2/Const*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*

index_type0*1
_output_shapes
:џџџџџџџџџИа 

=training/SGD/gradients/dropout_1/cond/Switch_1_grad/cond_gradMerge:training/SGD/gradients/dropout_1/cond/Merge_grad/cond_gradtraining/SGD/gradients/zeros_2**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџИа : *
T0
Щ
<training/SGD/gradients/dropout_1/cond/dropout/mul_grad/ShapeShapedropout_1/cond/dropout/truediv*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0*
_output_shapes
:
Щ
>training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Shape_1Shapedropout_1/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0*
_output_shapes
:
Я
Ltraining/SGD/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Shape>training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul

:training/SGD/gradients/dropout_1/cond/dropout/mul_grad/MulMul<training/SGD/gradients/dropout_1/cond/Merge_grad/cond_grad:1dropout_1/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџИа 
К
:training/SGD/gradients/dropout_1/cond/dropout/mul_grad/SumSum:training/SGD/gradients/dropout_1/cond/dropout/mul_grad/MulLtraining/SGD/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
М
>training/SGD/gradients/dropout_1/cond/dropout/mul_grad/ReshapeReshape:training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Sum<training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Shape*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0*1
_output_shapes
:џџџџџџџџџИа 

<training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Muldropout_1/cond/dropout/truediv<training/SGD/gradients/dropout_1/cond/Merge_grad/cond_grad:1*-
_class#
!loc:@dropout_1/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџИа *
T0
Р
<training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Sum<training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Ntraining/SGD/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Т
@training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape<training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Sum_1>training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0*1
_output_shapes
:џџџџџџџџџИа 
Х
@training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/ShapeShapedropout_1/cond/mul*
_output_shapes
:*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
out_type0
И
Btraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1Const*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
valueB *
dtype0*
_output_shapes
: 
п
Ptraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs@training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/ShapeBtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Btraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/RealDivRealDiv>training/SGD/gradients/dropout_1/cond/dropout/mul_grad/Reshapedropout_1/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџИа *
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
Ю
>training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/SumSumBtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/RealDivPtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgs*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0
Ь
Btraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/ReshapeReshape>training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Sum@training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Shape*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
Tshape0*1
_output_shapes
:џџџџџџџџџИа 
Ш
>training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/NegNegdropout_1/cond/mul*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа 

Dtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1RealDiv>training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Negdropout_1/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа 
 
Dtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2RealDivDtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1dropout_1/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа 
К
>training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/mulMul>training/SGD/gradients/dropout_1/cond/dropout/mul_grad/ReshapeDtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2*1
_output_shapes
:џџџџџџџџџИа *
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
Ю
@training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1Sum>training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/mulRtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0
З
Dtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Reshape_1Reshape@training/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1Btraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
Tshape0*
_output_shapes
: 
Ж
4training/SGD/gradients/dropout_1/cond/mul_grad/ShapeShapedropout_1/cond/mul/Switch:1*
T0*%
_class
loc:@dropout_1/cond/mul*
out_type0*
_output_shapes
:
 
6training/SGD/gradients/dropout_1/cond/mul_grad/Shape_1Const*%
_class
loc:@dropout_1/cond/mul*
valueB *
dtype0*
_output_shapes
: 
Џ
Dtraining/SGD/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4training/SGD/gradients/dropout_1/cond/mul_grad/Shape6training/SGD/gradients/dropout_1/cond/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@dropout_1/cond/mul
і
2training/SGD/gradients/dropout_1/cond/mul_grad/MulMulBtraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Reshapedropout_1/cond/mul/y*
T0*%
_class
loc:@dropout_1/cond/mul*1
_output_shapes
:џџџџџџџџџИа 

2training/SGD/gradients/dropout_1/cond/mul_grad/SumSum2training/SGD/gradients/dropout_1/cond/mul_grad/MulDtraining/SGD/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@dropout_1/cond/mul

6training/SGD/gradients/dropout_1/cond/mul_grad/ReshapeReshape2training/SGD/gradients/dropout_1/cond/mul_grad/Sum4training/SGD/gradients/dropout_1/cond/mul_grad/Shape*
T0*%
_class
loc:@dropout_1/cond/mul*
Tshape0*1
_output_shapes
:џџџџџџџџџИа 
џ
4training/SGD/gradients/dropout_1/cond/mul_grad/Mul_1Muldropout_1/cond/mul/Switch:1Btraining/SGD/gradients/dropout_1/cond/dropout/truediv_grad/Reshape*
T0*%
_class
loc:@dropout_1/cond/mul*1
_output_shapes
:џџџџџџџџџИа 
 
4training/SGD/gradients/dropout_1/cond/mul_grad/Sum_1Sum4training/SGD/gradients/dropout_1/cond/mul_grad/Mul_1Ftraining/SGD/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0

8training/SGD/gradients/dropout_1/cond/mul_grad/Reshape_1Reshape4training/SGD/gradients/dropout_1/cond/mul_grad/Sum_16training/SGD/gradients/dropout_1/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_1/cond/mul*
Tshape0*
_output_shapes
: 
п
training/SGD/gradients/Switch_3Switchleaky_re_lu_1/LeakyReludropout_1/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа 
Ж
!training/SGD/gradients/Identity_3Identitytraining/SGD/gradients/Switch_3*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*1
_output_shapes
:џџџџџџџџџИа 
Љ
training/SGD/gradients/Shape_4Shapetraining/SGD/gradients/Switch_3*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
out_type0*
_output_shapes
:
Й
$training/SGD/gradients/zeros_3/ConstConst"^training/SGD/gradients/Identity_3*
_output_shapes
: **
_class 
loc:@leaky_re_lu_1/LeakyRelu*
valueB
 *    *
dtype0
ц
training/SGD/gradients/zeros_3Filltraining/SGD/gradients/Shape_4$training/SGD/gradients/zeros_3/Const*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*

index_type0*1
_output_shapes
:џџџџџџџџџИа 

?training/SGD/gradients/dropout_1/cond/mul/Switch_grad/cond_gradMergetraining/SGD/gradients/zeros_36training/SGD/gradients/dropout_1/cond/mul_grad/Reshape*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџИа : 

training/SGD/gradients/AddN_1AddN=training/SGD/gradients/dropout_1/cond/Switch_1_grad/cond_grad?training/SGD/gradients/dropout_1/cond/mul/Switch_grad/cond_grad*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
N*1
_output_shapes
:џџџџџџџџџИа 
ј
Atraining/SGD/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGradLeakyReluGradtraining/SGD/gradients/AddN_1input/BiasAdd*1
_output_shapes
:џџџџџџџџџИа *
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
alpha%>
х
5training/SGD/gradients/input/BiasAdd_grad/BiasAddGradBiasAddGradAtraining/SGD/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
T0* 
_class
loc:@input/BiasAdd*
data_formatNHWC*
_output_shapes
: 
Ш
4training/SGD/gradients/input/convolution_grad/ShapeNShapeNinput_inputinput/kernel/read*$
_class
loc:@input/convolution*
out_type0*
N* 
_output_shapes
::*
T0
Д
Atraining/SGD/gradients/input/convolution_grad/Conv2DBackpropInputConv2DBackpropInput4training/SGD/gradients/input/convolution_grad/ShapeNinput/kernel/readAtraining/SGD/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа*
	dilations
*
T0*$
_class
loc:@input/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
Ї
Btraining/SGD/gradients/input/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterinput_input6training/SGD/gradients/input/convolution_grad/ShapeN:1Atraining/SGD/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*&
_output_shapes
: *
	dilations
*
T0*$
_class
loc:@input/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
^
training/SGD/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Ј
training/SGD/AssignAdd	AssignAddSGD/iterationstraining/SGD/AssignAdd/value*
T0	*!
_class
loc:@SGD/iterations*
_output_shapes
: *
use_locking( 
n
training/SGD/CastCastSGD/iterations/read*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0	
[
training/SGD/mulMulSGD/decay/readtraining/SGD/Cast*
T0*
_output_shapes
: 
W
training/SGD/add/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
^
training/SGD/addAddtraining/SGD/add/xtraining/SGD/mul*
T0*
_output_shapes
: 
[
training/SGD/truediv/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
j
training/SGD/truedivRealDivtraining/SGD/truediv/xtraining/SGD/add*
T0*
_output_shapes
: 
]
training/SGD/mul_1MulSGD/lr/readtraining/SGD/truediv*
_output_shapes
: *
T0
w
training/SGD/zerosConst*%
valueB *    *
dtype0*&
_output_shapes
: 

training/SGD/Variable
VariableV2*
shape: *
shared_name *
dtype0*
	container *&
_output_shapes
: 
е
training/SGD/Variable/AssignAssigntraining/SGD/Variabletraining/SGD/zeros*
use_locking(*
T0*(
_class
loc:@training/SGD/Variable*
validate_shape(*&
_output_shapes
: 

training/SGD/Variable/readIdentitytraining/SGD/Variable*
T0*(
_class
loc:@training/SGD/Variable*&
_output_shapes
: 
a
training/SGD/zeros_1Const*
valueB *    *
dtype0*
_output_shapes
: 

training/SGD/Variable_1
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
shape: 
б
training/SGD/Variable_1/AssignAssigntraining/SGD/Variable_1training/SGD/zeros_1*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_1*
validate_shape(*
_output_shapes
: 

training/SGD/Variable_1/readIdentitytraining/SGD/Variable_1*
T0**
_class 
loc:@training/SGD/Variable_1*
_output_shapes
: 
}
$training/SGD/zeros_2/shape_as_tensorConst*%
valueB"          @   *
dtype0*
_output_shapes
:
_
training/SGD/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ё
training/SGD/zeros_2Fill$training/SGD/zeros_2/shape_as_tensortraining/SGD/zeros_2/Const*

index_type0*&
_output_shapes
: @*
T0

training/SGD/Variable_2
VariableV2*
	container *&
_output_shapes
: @*
shape: @*
shared_name *
dtype0
н
training/SGD/Variable_2/AssignAssigntraining/SGD/Variable_2training/SGD/zeros_2*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_2*
validate_shape(*&
_output_shapes
: @

training/SGD/Variable_2/readIdentitytraining/SGD/Variable_2*&
_output_shapes
: @*
T0**
_class 
loc:@training/SGD/Variable_2
a
training/SGD/zeros_3Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/SGD/Variable_3
VariableV2*
	container *
_output_shapes
:@*
shape:@*
shared_name *
dtype0
б
training/SGD/Variable_3/AssignAssigntraining/SGD/Variable_3training/SGD/zeros_3*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_3*
validate_shape(*
_output_shapes
:@

training/SGD/Variable_3/readIdentitytraining/SGD/Variable_3*
_output_shapes
:@*
T0**
_class 
loc:@training/SGD/Variable_3
}
$training/SGD/zeros_4/shape_as_tensorConst*%
valueB"      @      *
dtype0*
_output_shapes
:
_
training/SGD/zeros_4/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ё
training/SGD/zeros_4Fill$training/SGD/zeros_4/shape_as_tensortraining/SGD/zeros_4/Const*
T0*

index_type0*&
_output_shapes
:@

training/SGD/Variable_4
VariableV2*
dtype0*
	container *&
_output_shapes
:@*
shape:@*
shared_name 
н
training/SGD/Variable_4/AssignAssigntraining/SGD/Variable_4training/SGD/zeros_4*&
_output_shapes
:@*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_4*
validate_shape(

training/SGD/Variable_4/readIdentitytraining/SGD/Variable_4*
T0**
_class 
loc:@training/SGD/Variable_4*&
_output_shapes
:@
a
training/SGD/zeros_5Const*
valueB*    *
dtype0*
_output_shapes
:

training/SGD/Variable_5
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
б
training/SGD/Variable_5/AssignAssigntraining/SGD/Variable_5training/SGD/zeros_5*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_5*
validate_shape(*
_output_shapes
:

training/SGD/Variable_5/readIdentitytraining/SGD/Variable_5*
_output_shapes
:*
T0**
_class 
loc:@training/SGD/Variable_5
y
training/SGD/mul_2MulSGD/momentum/readtraining/SGD/Variable/read*
T0*&
_output_shapes
: 
Ђ
training/SGD/mul_3Multraining/SGD/mul_1Btraining/SGD/gradients/input/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: *
T0
p
training/SGD/subSubtraining/SGD/mul_2training/SGD/mul_3*
T0*&
_output_shapes
: 
Ъ
training/SGD/AssignAssigntraining/SGD/Variabletraining/SGD/sub*
T0*(
_class
loc:@training/SGD/Variable*
validate_shape(*&
_output_shapes
: *
use_locking(
o
training/SGD/mul_4MulSGD/momentum/readtraining/SGD/sub*
T0*&
_output_shapes
: 
q
training/SGD/add_1Addinput/kernel/readtraining/SGD/mul_4*
T0*&
_output_shapes
: 
Ђ
training/SGD/mul_5Multraining/SGD/mul_1Btraining/SGD/gradients/input/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
r
training/SGD/sub_1Subtraining/SGD/add_1training/SGD/mul_5*
T0*&
_output_shapes
: 
М
training/SGD/Assign_1Assigninput/kerneltraining/SGD/sub_1*
use_locking(*
T0*
_class
loc:@input/kernel*
validate_shape(*&
_output_shapes
: 
o
training/SGD/mul_6MulSGD/momentum/readtraining/SGD/Variable_1/read*
_output_shapes
: *
T0

training/SGD/mul_7Multraining/SGD/mul_15training/SGD/gradients/input/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
f
training/SGD/sub_2Subtraining/SGD/mul_6training/SGD/mul_7*
T0*
_output_shapes
: 
Ц
training/SGD/Assign_2Assigntraining/SGD/Variable_1training/SGD/sub_2*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_1*
validate_shape(*
_output_shapes
: 
e
training/SGD/mul_8MulSGD/momentum/readtraining/SGD/sub_2*
_output_shapes
: *
T0
c
training/SGD/add_2Addinput/bias/readtraining/SGD/mul_8*
_output_shapes
: *
T0

training/SGD/mul_9Multraining/SGD/mul_15training/SGD/gradients/input/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
f
training/SGD/sub_3Subtraining/SGD/add_2training/SGD/mul_9*
_output_shapes
: *
T0
Ќ
training/SGD/Assign_3Assign
input/biastraining/SGD/sub_3*
use_locking(*
T0*
_class
loc:@input/bias*
validate_shape(*
_output_shapes
: 
|
training/SGD/mul_10MulSGD/momentum/readtraining/SGD/Variable_2/read*
T0*&
_output_shapes
: @
І
training/SGD/mul_11Multraining/SGD/mul_1Etraining/SGD/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
t
training/SGD/sub_4Subtraining/SGD/mul_10training/SGD/mul_11*&
_output_shapes
: @*
T0
в
training/SGD/Assign_4Assigntraining/SGD/Variable_2training/SGD/sub_4*&
_output_shapes
: @*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_2*
validate_shape(
r
training/SGD/mul_12MulSGD/momentum/readtraining/SGD/sub_4*
T0*&
_output_shapes
: @
u
training/SGD/add_3Addconv2d_1/kernel/readtraining/SGD/mul_12*
T0*&
_output_shapes
: @
І
training/SGD/mul_13Multraining/SGD/mul_1Etraining/SGD/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
s
training/SGD/sub_5Subtraining/SGD/add_3training/SGD/mul_13*&
_output_shapes
: @*
T0
Т
training/SGD/Assign_5Assignconv2d_1/kerneltraining/SGD/sub_5*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel
p
training/SGD/mul_14MulSGD/momentum/readtraining/SGD/Variable_3/read*
_output_shapes
:@*
T0

training/SGD/mul_15Multraining/SGD/mul_18training/SGD/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
h
training/SGD/sub_6Subtraining/SGD/mul_14training/SGD/mul_15*
T0*
_output_shapes
:@
Ц
training/SGD/Assign_6Assigntraining/SGD/Variable_3training/SGD/sub_6*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_3
f
training/SGD/mul_16MulSGD/momentum/readtraining/SGD/sub_6*
_output_shapes
:@*
T0
g
training/SGD/add_4Addconv2d_1/bias/readtraining/SGD/mul_16*
T0*
_output_shapes
:@

training/SGD/mul_17Multraining/SGD/mul_18training/SGD/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
g
training/SGD/sub_7Subtraining/SGD/add_4training/SGD/mul_17*
T0*
_output_shapes
:@
В
training/SGD/Assign_7Assignconv2d_1/biastraining/SGD/sub_7*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:@
|
training/SGD/mul_18MulSGD/momentum/readtraining/SGD/Variable_4/read*&
_output_shapes
:@*
T0
Є
training/SGD/mul_19Multraining/SGD/mul_1Ctraining/SGD/gradients/output/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
t
training/SGD/sub_8Subtraining/SGD/mul_18training/SGD/mul_19*
T0*&
_output_shapes
:@
в
training/SGD/Assign_8Assigntraining/SGD/Variable_4training/SGD/sub_8**
_class 
loc:@training/SGD/Variable_4*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
r
training/SGD/mul_20MulSGD/momentum/readtraining/SGD/sub_8*&
_output_shapes
:@*
T0
s
training/SGD/add_5Addoutput/kernel/readtraining/SGD/mul_20*
T0*&
_output_shapes
:@
Є
training/SGD/mul_21Multraining/SGD/mul_1Ctraining/SGD/gradients/output/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
s
training/SGD/sub_9Subtraining/SGD/add_5training/SGD/mul_21*
T0*&
_output_shapes
:@
О
training/SGD/Assign_9Assignoutput/kerneltraining/SGD/sub_9*&
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@output/kernel*
validate_shape(
p
training/SGD/mul_22MulSGD/momentum/readtraining/SGD/Variable_5/read*
_output_shapes
:*
T0

training/SGD/mul_23Multraining/SGD/mul_16training/SGD/gradients/output/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
i
training/SGD/sub_10Subtraining/SGD/mul_22training/SGD/mul_23*
T0*
_output_shapes
:
Ш
training/SGD/Assign_10Assigntraining/SGD/Variable_5training/SGD/sub_10**
_class 
loc:@training/SGD/Variable_5*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
g
training/SGD/mul_24MulSGD/momentum/readtraining/SGD/sub_10*
_output_shapes
:*
T0
e
training/SGD/add_6Addoutput/bias/readtraining/SGD/mul_24*
T0*
_output_shapes
:

training/SGD/mul_25Multraining/SGD/mul_16training/SGD/gradients/output/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
h
training/SGD/sub_11Subtraining/SGD/add_6training/SGD/mul_25*
T0*
_output_shapes
:
А
training/SGD/Assign_11Assignoutput/biastraining/SGD/sub_11*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@output/bias
п
training/group_depsNoOp	^loss/mul^training/SGD/Assign^training/SGD/AssignAdd^training/SGD/Assign_1^training/SGD/Assign_10^training/SGD/Assign_11^training/SGD/Assign_2^training/SGD/Assign_3^training/SGD/Assign_4^training/SGD/Assign_5^training/SGD/Assign_6^training/SGD/Assign_7^training/SGD/Assign_8^training/SGD/Assign_9

IsVariableInitializedIsVariableInitializedinput/kernel*
_class
loc:@input/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitialized
input/bias*
_class
loc:@input/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_2IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_3IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializedoutput/kernel* 
_class
loc:@output/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializedoutput/bias*
dtype0*
_output_shapes
: *
_class
loc:@output/bias

IsVariableInitialized_6IsVariableInitializedSGD/iterations*!
_class
loc:@SGD/iterations*
dtype0	*
_output_shapes
: 
x
IsVariableInitialized_7IsVariableInitializedSGD/lr*
_class
loc:@SGD/lr*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializedSGD/momentum*
dtype0*
_output_shapes
: *
_class
loc:@SGD/momentum
~
IsVariableInitialized_9IsVariableInitialized	SGD/decay*
_class
loc:@SGD/decay*
dtype0*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitializedtraining/SGD/Variable*
_output_shapes
: *(
_class
loc:@training/SGD/Variable*
dtype0

IsVariableInitialized_11IsVariableInitializedtraining/SGD/Variable_1**
_class 
loc:@training/SGD/Variable_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_12IsVariableInitializedtraining/SGD/Variable_2**
_class 
loc:@training/SGD/Variable_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_13IsVariableInitializedtraining/SGD/Variable_3*
_output_shapes
: **
_class 
loc:@training/SGD/Variable_3*
dtype0

IsVariableInitialized_14IsVariableInitializedtraining/SGD/Variable_4**
_class 
loc:@training/SGD/Variable_4*
dtype0*
_output_shapes
: 

IsVariableInitialized_15IsVariableInitializedtraining/SGD/Variable_5**
_class 
loc:@training/SGD/Variable_5*
dtype0*
_output_shapes
: 
Ї
initNoOp^SGD/decay/Assign^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^input/bias/Assign^input/kernel/Assign^output/bias/Assign^output/kernel/Assign^training/SGD/Variable/Assign^training/SGD/Variable_1/Assign^training/SGD/Variable_2/Assign^training/SGD/Variable_3/Assign^training/SGD/Variable_4/Assign^training/SGD/Variable_5/Assign""Ј
trainable_variables
T
input/kernel:0input/kernel/Assigninput/kernel/read:02input/random_uniform:08
E
input/bias:0input/bias/Assigninput/bias/read:02input/Const:08
`
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02conv2d_1/random_uniform:08
Q
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02conv2d_1/Const:08
X
output/kernel:0output/kernel/Assignoutput/kernel/read:02output/random_uniform:08
I
output/bias:0output/bias/Assignoutput/bias/read:02output/Const:08
b
SGD/iterations:0SGD/iterations/AssignSGD/iterations/read:02SGD/iterations/initial_value:08
B
SGD/lr:0SGD/lr/AssignSGD/lr/read:02SGD/lr/initial_value:08
Z
SGD/momentum:0SGD/momentum/AssignSGD/momentum/read:02SGD/momentum/initial_value:08
N
SGD/decay:0SGD/decay/AssignSGD/decay/read:02SGD/decay/initial_value:08
m
training/SGD/Variable:0training/SGD/Variable/Assigntraining/SGD/Variable/read:02training/SGD/zeros:08
u
training/SGD/Variable_1:0training/SGD/Variable_1/Assigntraining/SGD/Variable_1/read:02training/SGD/zeros_1:08
u
training/SGD/Variable_2:0training/SGD/Variable_2/Assigntraining/SGD/Variable_2/read:02training/SGD/zeros_2:08
u
training/SGD/Variable_3:0training/SGD/Variable_3/Assigntraining/SGD/Variable_3/read:02training/SGD/zeros_3:08
u
training/SGD/Variable_4:0training/SGD/Variable_4/Assigntraining/SGD/Variable_4/read:02training/SGD/zeros_4:08
u
training/SGD/Variable_5:0training/SGD/Variable_5/Assigntraining/SGD/Variable_5/read:02training/SGD/zeros_5:08"
cond_contextћј
ю
dropout_1/cond/cond_textdropout_1/cond/pred_id:0dropout_1/cond/switch_t:0 *
dropout_1/cond/dropout/Floor:0
dropout_1/cond/dropout/Shape:0
dropout_1/cond/dropout/add:0
dropout_1/cond/dropout/mul:0
5dropout_1/cond/dropout/random_uniform/RandomUniform:0
+dropout_1/cond/dropout/random_uniform/max:0
+dropout_1/cond/dropout/random_uniform/min:0
+dropout_1/cond/dropout/random_uniform/mul:0
+dropout_1/cond/dropout/random_uniform/sub:0
'dropout_1/cond/dropout/random_uniform:0
dropout_1/cond/dropout/rate:0
dropout_1/cond/dropout/sub/x:0
dropout_1/cond/dropout/sub:0
 dropout_1/cond/dropout/truediv:0
dropout_1/cond/mul/Switch:1
dropout_1/cond/mul/y:0
dropout_1/cond/mul:0
dropout_1/cond/pred_id:0
dropout_1/cond/switch_t:0
leaky_re_lu_1/LeakyRelu:04
dropout_1/cond/pred_id:0dropout_1/cond/pred_id:08
leaky_re_lu_1/LeakyRelu:0dropout_1/cond/mul/Switch:1
Ш
dropout_1/cond/cond_text_1dropout_1/cond/pred_id:0dropout_1/cond/switch_f:0*є
dropout_1/cond/Switch_1:0
dropout_1/cond/Switch_1:1
dropout_1/cond/pred_id:0
dropout_1/cond/switch_f:0
leaky_re_lu_1/LeakyRelu:04
dropout_1/cond/pred_id:0dropout_1/cond/pred_id:06
leaky_re_lu_1/LeakyRelu:0dropout_1/cond/Switch_1:0
ю
dropout_2/cond/cond_textdropout_2/cond/pred_id:0dropout_2/cond/switch_t:0 *
dropout_2/cond/dropout/Floor:0
dropout_2/cond/dropout/Shape:0
dropout_2/cond/dropout/add:0
dropout_2/cond/dropout/mul:0
5dropout_2/cond/dropout/random_uniform/RandomUniform:0
+dropout_2/cond/dropout/random_uniform/max:0
+dropout_2/cond/dropout/random_uniform/min:0
+dropout_2/cond/dropout/random_uniform/mul:0
+dropout_2/cond/dropout/random_uniform/sub:0
'dropout_2/cond/dropout/random_uniform:0
dropout_2/cond/dropout/rate:0
dropout_2/cond/dropout/sub/x:0
dropout_2/cond/dropout/sub:0
 dropout_2/cond/dropout/truediv:0
dropout_2/cond/mul/Switch:1
dropout_2/cond/mul/y:0
dropout_2/cond/mul:0
dropout_2/cond/pred_id:0
dropout_2/cond/switch_t:0
leaky_re_lu_2/LeakyRelu:04
dropout_2/cond/pred_id:0dropout_2/cond/pred_id:08
leaky_re_lu_2/LeakyRelu:0dropout_2/cond/mul/Switch:1
Ш
dropout_2/cond/cond_text_1dropout_2/cond/pred_id:0dropout_2/cond/switch_f:0*є
dropout_2/cond/Switch_1:0
dropout_2/cond/Switch_1:1
dropout_2/cond/pred_id:0
dropout_2/cond/switch_f:0
leaky_re_lu_2/LeakyRelu:06
leaky_re_lu_2/LeakyRelu:0dropout_2/cond/Switch_1:04
dropout_2/cond/pred_id:0dropout_2/cond/pred_id:0"
	variables
T
input/kernel:0input/kernel/Assigninput/kernel/read:02input/random_uniform:08
E
input/bias:0input/bias/Assigninput/bias/read:02input/Const:08
`
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02conv2d_1/random_uniform:08
Q
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02conv2d_1/Const:08
X
output/kernel:0output/kernel/Assignoutput/kernel/read:02output/random_uniform:08
I
output/bias:0output/bias/Assignoutput/bias/read:02output/Const:08
b
SGD/iterations:0SGD/iterations/AssignSGD/iterations/read:02SGD/iterations/initial_value:08
B
SGD/lr:0SGD/lr/AssignSGD/lr/read:02SGD/lr/initial_value:08
Z
SGD/momentum:0SGD/momentum/AssignSGD/momentum/read:02SGD/momentum/initial_value:08
N
SGD/decay:0SGD/decay/AssignSGD/decay/read:02SGD/decay/initial_value:08
m
training/SGD/Variable:0training/SGD/Variable/Assigntraining/SGD/Variable/read:02training/SGD/zeros:08
u
training/SGD/Variable_1:0training/SGD/Variable_1/Assigntraining/SGD/Variable_1/read:02training/SGD/zeros_1:08
u
training/SGD/Variable_2:0training/SGD/Variable_2/Assigntraining/SGD/Variable_2/read:02training/SGD/zeros_2:08
u
training/SGD/Variable_3:0training/SGD/Variable_3/Assigntraining/SGD/Variable_3/read:02training/SGD/zeros_3:08
u
training/SGD/Variable_4:0training/SGD/Variable_4/Assigntraining/SGD/Variable_4/read:02training/SGD/zeros_4:08
u
training/SGD/Variable_5:0training/SGD/Variable_5/Assigntraining/SGD/Variable_5/read:02training/SGD/zeros_5:08O	@       ЃK"	kъ`і(зA*

lossђ>ў$M       и-	мЙШdі(зA*

loss;zё>ЛELO       и-	­1gі(зA*

lossоY№>oГDз       и-	}Бtjі(зA*

lossD7я>pНј       и-	'ЃBmі(зA*

lossuю>Ш)c       и-	XVpі(зA*

lossFѓь>@MіЦ       и-	t_ёrі(зA*

lossбы>ЮШ­       и-	эuі(зA*

lossВъ>П8К       и-	ХуУxі(зA*

lossщ>С$       и-	BА{і(зA	*

lossDш>	.u       и-	џъ~~і(зA
*

lossч>"ў5       и-	``і(зA*

lossц>еѓѕ       и-	tb?і(зA*

lossUх>v­Ђ