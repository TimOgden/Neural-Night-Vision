       ЃK"	   VК зAbrain.Event:2Bъ)VХ     ;ХЉ	|*!VК зA"Щ

conv2d_1_inputPlaceholder*1
_output_shapes
:џџџџџџџџџИа*&
shape:џџџџџџџџџИа*
dtype0
v
conv2d_1/random_uniform/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
`
conv2d_1/random_uniform/minConst*
valueB
 *OSО*
dtype0*
_output_shapes
: 
`
conv2d_1/random_uniform/maxConst*
valueB
 *OS>*
dtype0*
_output_shapes
: 
В
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
seedБџх)*
T0*
dtype0*&
_output_shapes
: *
seed2ГЄ
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
T0*
_output_shapes
: 

conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*&
_output_shapes
: *
T0

conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*&
_output_shapes
: *
T0

conv2d_1/kernel
VariableV2*
dtype0*&
_output_shapes
: *
	container *
shape: *
shared_name 
Ш
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*&
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(

conv2d_1/kernel/readIdentityconv2d_1/kernel*&
_output_shapes
: *
T0*"
_class
loc:@conv2d_1/kernel
[
conv2d_1/ConstConst*
valueB *    *
dtype0*
_output_shapes
: 
y
conv2d_1/bias
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
­
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(
t
conv2d_1/bias/readIdentityconv2d_1/bias*
_output_shapes
: *
T0* 
_class
loc:@conv2d_1/bias
s
"conv2d_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ю
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*1
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

conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа 

leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_1/BiasAdd*
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
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
_output_shapes
: *
T0

c
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
: *
T0

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
dropout_1/cond/dropout/rateConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ=
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
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
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0
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
seed2кл*
seedБџх)
Ї
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ь
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*1
_output_shapes
:џџџџџџџџџИа 
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
dropout_1/cond/dropout/truedivRealDivdropout_1/cond/muldropout_1/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџИа *
T0
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
dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*3
_output_shapes!
:џџџџџџџџџИа : *
T0*
N
Ц
max_pooling2d_1/MaxPoolMaxPooldropout_1/cond/Merge*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ *
T0*
data_formatNHWC*
strides

v
conv2d_2/random_uniform/shapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
`
conv2d_2/random_uniform/minConst*
valueB
 *ЋЊЊН*
dtype0*
_output_shapes
: 
`
conv2d_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ЋЊЊ=
В
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
T0*
dtype0*&
_output_shapes
: @*
seed2ђшр*
seedБџх)
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
T0*
_output_shapes
: 

conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*&
_output_shapes
: @*
T0

conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*&
_output_shapes
: @*
T0

conv2d_2/kernel
VariableV2*
shape: @*
shared_name *
dtype0*&
_output_shapes
: @*
	container 
Ш
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
: @*
use_locking(

conv2d_2/kernel/readIdentityconv2d_2/kernel*&
_output_shapes
: @*
T0*"
_class
loc:@conv2d_2/kernel
[
conv2d_2/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_2/bias
VariableV2*
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
­
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@
t
conv2d_2/bias/readIdentityconv2d_2/bias*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ї
conv2d_2/convolutionConv2Dmax_pooling2d_1/MaxPoolconv2d_2/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ@

conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџЈ@*
T0

leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_2/BiasAdd*
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
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
T0
*
_output_shapes
: 
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
dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*
T0*1
_output_shapes
:џџџџџџџџџЈ@
й
dropout_2/cond/mul/SwitchSwitchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@
z
dropout_2/cond/dropout/rateConst^dropout_2/cond/switch_t*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
_output_shapes
:*
T0*
out_type0
{
dropout_2/cond/dropout/sub/xConst^dropout_2/cond/switch_t*
_output_shapes
: *
valueB
 *  ?*
dtype0
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
seedБџх)*
T0*
dtype0*1
_output_shapes
:џџџџџџџџџЈ@*
seed2Пј
Ї
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
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
dropout_2/cond/Switch_1Switchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@

dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
N*3
_output_shapes!
:џџџџџџџџџЈ@: *
T0
i
up_sampling2d_1/ShapeShapedropout_2/cond/Merge*
out_type0*
_output_shapes
:*
T0
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
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape#up_sampling2d_1/strided_slice/stack%up_sampling2d_1/strided_slice/stack_1%up_sampling2d_1/strided_slice/stack_2*
T0*
Index0*
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
v
conv2d_3/random_uniform/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
`
conv2d_3/random_uniform/minConst*
valueB
 *8JЬН*
dtype0*
_output_shapes
: 
`
conv2d_3/random_uniform/maxConst*
valueB
 *8JЬ=*
dtype0*
_output_shapes
: 
В
%conv2d_3/random_uniform/RandomUniformRandomUniformconv2d_3/random_uniform/shape*
dtype0*&
_output_shapes
:@*
seed2§щ*
seedБџх)*
T0
}
conv2d_3/random_uniform/subSubconv2d_3/random_uniform/maxconv2d_3/random_uniform/min*
_output_shapes
: *
T0

conv2d_3/random_uniform/mulMul%conv2d_3/random_uniform/RandomUniformconv2d_3/random_uniform/sub*&
_output_shapes
:@*
T0

conv2d_3/random_uniformAddconv2d_3/random_uniform/mulconv2d_3/random_uniform/min*
T0*&
_output_shapes
:@

conv2d_3/kernel
VariableV2*&
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
Ш
conv2d_3/kernel/AssignAssignconv2d_3/kernelconv2d_3/random_uniform*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel

conv2d_3/kernel/readIdentityconv2d_3/kernel*&
_output_shapes
:@*
T0*"
_class
loc:@conv2d_3/kernel
[
conv2d_3/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv2d_3/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
­
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/Const*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes
:
t
conv2d_3/bias/readIdentityconv2d_3/bias*
T0* 
_class
loc:@conv2d_3/bias*
_output_shapes
:
s
"conv2d_3/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0

conv2d_3/convolutionConv2D%up_sampling2d_1/ResizeNearestNeighborconv2d_3/kernel/read*
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

conv2d_3/BiasAddBiasAddconv2d_3/convolutionconv2d_3/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа
_
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
dtype0	*
_output_shapes
: *
	container *
shape: *
shared_name 
О
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
use_locking(*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: 
v
Adam/iterations/readIdentityAdam/iterations*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
Z
Adam/lr/initial_valueConst*
valueB
 *Зб8*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 

Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
use_locking(*
T0*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: 
^
Adam/lr/readIdentityAdam/lr*
_output_shapes
: *
T0*
_class
loc:@Adam/lr
^
Adam/beta_1/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
Adam/beta_1
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ў
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: 
j
Adam/beta_1/readIdentityAdam/beta_1*
_output_shapes
: *
T0*
_class
loc:@Adam/beta_1
^
Adam/beta_2/initial_valueConst*
valueB
 *wО?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
Ў
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_2*
validate_shape(*
_output_shapes
: 
j
Adam/beta_2/readIdentityAdam/beta_2*
_output_shapes
: *
T0*
_class
loc:@Adam/beta_2
]
Adam/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

Adam/decay
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
Њ
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/decay
g
Adam/decay/readIdentity
Adam/decay*
_output_shapes
: *
T0*
_class
loc:@Adam/decay
И
conv2d_3_targetPlaceholder*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*?
shape6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0
r
conv2d_3_sample_weightsPlaceholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
|
loss/conv2d_3_loss/subSubconv2d_3/BiasAddconv2d_3_target*
T0*1
_output_shapes
:џџџџџџџџџИа
q
loss/conv2d_3_loss/AbsAbsloss/conv2d_3_loss/sub*
T0*1
_output_shapes
:џџџџџџџџџИа
t
)loss/conv2d_3_loss/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
З
loss/conv2d_3_loss/MeanMeanloss/conv2d_3_loss/Abs)loss/conv2d_3_loss/Mean/reduction_indices*-
_output_shapes
:џџџџџџџџџИа*
	keep_dims( *

Tidx0*
T0
|
+loss/conv2d_3_loss/Mean_1/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
В
loss/conv2d_3_loss/Mean_1Meanloss/conv2d_3_loss/Mean+loss/conv2d_3_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0

loss/conv2d_3_loss/mulMulloss/conv2d_3_loss/Mean_1conv2d_3_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
b
loss/conv2d_3_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/conv2d_3_loss/NotEqualNotEqualconv2d_3_sample_weightsloss/conv2d_3_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ

loss/conv2d_3_loss/CastCastloss/conv2d_3_loss/NotEqual*

SrcT0
*
Truncate( *#
_output_shapes
:џџџџџџџџџ*

DstT0
b
loss/conv2d_3_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/conv2d_3_loss/Mean_2Meanloss/conv2d_3_loss/Castloss/conv2d_3_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 

loss/conv2d_3_loss/truedivRealDivloss/conv2d_3_loss/mulloss/conv2d_3_loss/Mean_2*
T0*#
_output_shapes
:џџџџџџџџџ
d
loss/conv2d_3_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/conv2d_3_loss/Mean_3Meanloss/conv2d_3_loss/truedivloss/conv2d_3_loss/Const_1*
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
W
loss/mulMul
loss/mul/xloss/conv2d_3_loss/Mean_3*
_output_shapes
: *
T0
}
training/Adam/gradients/ShapeConst*
valueB *
_class
loc:@loss/mul*
dtype0*
_output_shapes
: 

!training/Adam/gradients/grad_ys_0Const*
valueB
 *  ?*
_class
loc:@loss/mul*
dtype0*
_output_shapes
: 
Ж
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*

index_type0*
_class
loc:@loss/mul*
_output_shapes
: 
Ї
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/conv2d_3_loss/Mean_3*
_class
loc:@loss/mul*
_output_shapes
: *
T0

+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
М
Dtraining/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3

>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Dtraining/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Reshape/shape*
T0*
Tshape0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
_output_shapes
:
Ф
<training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/ShapeShapeloss/conv2d_3_loss/truediv*
T0*
out_type0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
_output_shapes
:
Џ
;training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/TileTile>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Reshape<training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape*

Tmultiples0*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Ц
>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape_1Shapeloss/conv2d_3_loss/truediv*
_output_shapes
:*
T0*
out_type0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3
Џ
>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape_2Const*
valueB *,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
dtype0*
_output_shapes
: 
Д
<training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/ConstConst*
valueB: *,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
dtype0*
_output_shapes
:
­
;training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/ProdProd>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape_1<training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Const*

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
_output_shapes
: 
Ж
>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Const_1Const*
valueB: *,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
dtype0*
_output_shapes
:
Б
=training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Prod_1Prod>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape_2>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3
А
@training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3

>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/MaximumMaximum=training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Prod_1@training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Maximum/y*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
_output_shapes
: 

?training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/floordivFloorDiv;training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Prod>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Maximum*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
_output_shapes
: *
T0
ђ
;training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/CastCast?training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
Truncate( 

>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/truedivRealDiv;training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Tile;training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Cast*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ*
T0
Т
=training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/ShapeShapeloss/conv2d_3_loss/mul*
T0*
out_type0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*
_output_shapes
:
Б
?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape_1Const*
valueB *-
_class#
!loc:@loss/conv2d_3_loss/truediv*
dtype0*
_output_shapes
: 
в
Mtraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape_1*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDivRealDiv>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/truedivloss/conv2d_3_loss/Mean_2*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ
С
;training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/SumSum?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDivMtraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv
Б
?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/ReshapeReshape;training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Sum=training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0*-
_class#
!loc:@loss/conv2d_3_loss/truediv
З
;training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/NegNegloss/conv2d_3_loss/mul*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Atraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDiv_1RealDiv;training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Negloss/conv2d_3_loss/Mean_2*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Atraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDiv_2RealDivAtraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDiv_1loss/conv2d_3_loss/Mean_2*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ
Ђ
;training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/mulMul>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/truedivAtraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDiv_2*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ
С
=training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Sum_1Sum;training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/mulOtraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
Њ
Atraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Reshape_1Reshape=training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Sum_1?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape_1*
Tshape0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*
_output_shapes
: *
T0
Н
9training/Adam/gradients/loss/conv2d_3_loss/mul_grad/ShapeShapeloss/conv2d_3_loss/Mean_1*
T0*
out_type0*)
_class
loc:@loss/conv2d_3_loss/mul*
_output_shapes
:
Н
;training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape_1Shapeconv2d_3_sample_weights*
T0*
out_type0*)
_class
loc:@loss/conv2d_3_loss/mul*
_output_shapes
:
Т
Itraining/Adam/gradients/loss/conv2d_3_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape;training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape_1*
T0*)
_class
loc:@loss/conv2d_3_loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ё
7training/Adam/gradients/loss/conv2d_3_loss/mul_grad/MulMul?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Reshapeconv2d_3_sample_weights*
T0*)
_class
loc:@loss/conv2d_3_loss/mul*#
_output_shapes
:џџџџџџџџџ
­
7training/Adam/gradients/loss/conv2d_3_loss/mul_grad/SumSum7training/Adam/gradients/loss/conv2d_3_loss/mul_grad/MulItraining/Adam/gradients/loss/conv2d_3_loss/mul_grad/BroadcastGradientArgs*
T0*)
_class
loc:@loss/conv2d_3_loss/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ё
;training/Adam/gradients/loss/conv2d_3_loss/mul_grad/ReshapeReshape7training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Sum9training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape*
Tshape0*)
_class
loc:@loss/conv2d_3_loss/mul*#
_output_shapes
:џџџџџџџџџ*
T0
ѕ
9training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Mul_1Mulloss/conv2d_3_loss/Mean_1?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Reshape*#
_output_shapes
:џџџџџџџџџ*
T0*)
_class
loc:@loss/conv2d_3_loss/mul
Г
9training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Sum_1Sum9training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Mul_1Ktraining/Adam/gradients/loss/conv2d_3_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*)
_class
loc:@loss/conv2d_3_loss/mul
Ї
=training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Reshape_1Reshape9training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Sum_1;training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape_1*
T0*
Tshape0*)
_class
loc:@loss/conv2d_3_loss/mul*#
_output_shapes
:џџџџџџџџџ
С
<training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/ShapeShapeloss/conv2d_3_loss/Mean*
_output_shapes
:*
T0*
out_type0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1
Ћ
;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/SizeConst*
value	B :*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
dtype0*
_output_shapes
: 
ў
:training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/addAdd+loss/conv2d_3_loss/Mean_1/reduction_indices;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Size*
_output_shapes
:*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1

:training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/modFloorMod:training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/add;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Size*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
:
Ж
>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_1Const*
valueB:*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
dtype0*
_output_shapes
:
В
Btraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *,
_class"
 loc:@loss/conv2d_3_loss/Mean_1
В
Btraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range/deltaConst*
value	B :*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
dtype0*
_output_shapes
: 
р
<training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/rangeRangeBtraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range/start;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/SizeBtraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range/delta*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
:*

Tidx0
Б
Atraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1
Ћ
;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/FillFill>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_1Atraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Fill/value*

index_type0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
:*
T0
І
Dtraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/DynamicStitchDynamicStitch<training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range:training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/mod<training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Fill*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
N*
_output_shapes
:*
T0
А
@training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum/yConst*
value	B :*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
dtype0*
_output_shapes
: 
Є
>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/MaximumMaximumDtraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/DynamicStitch@training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum/y*
_output_shapes
:*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1

?training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/floordivFloorDiv<training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
:
а
>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/ReshapeReshape;training/Adam/gradients/loss/conv2d_3_loss/mul_grad/ReshapeDtraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/DynamicStitch*
T0*
Tshape0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ь
;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/TileTile>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Reshape?training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/floordiv*

Tmultiples0*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
У
>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_2Shapeloss/conv2d_3_loss/Mean*
out_type0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
:*
T0
Х
>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_3Shapeloss/conv2d_3_loss/Mean_1*
T0*
out_type0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
:
Д
<training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/ConstConst*
valueB: *,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
dtype0*
_output_shapes
:
­
;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/ProdProd>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_2<training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Const*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ж
>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Const_1Const*
valueB: *,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
dtype0*
_output_shapes
:
Б
=training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Prod_1Prod>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_3>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Const_1*

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
: 
В
Btraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum_1/yConst*
value	B :*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
dtype0*
_output_shapes
: 

@training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum_1Maximum=training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Prod_1Btraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum_1/y*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
: 

Atraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/floordiv_1FloorDiv;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Prod@training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum_1*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
: *
T0
є
;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/CastCastAtraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/floordiv_1*

SrcT0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
Truncate( *
_output_shapes
: *

DstT0
Љ
>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/truedivRealDiv;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Tile;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Cast*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*-
_output_shapes
:џџџџџџџџџИа
М
:training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/ShapeShapeloss/conv2d_3_loss/Abs*
_output_shapes
:*
T0*
out_type0**
_class 
loc:@loss/conv2d_3_loss/Mean
Ї
9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/SizeConst*
value	B :**
_class 
loc:@loss/conv2d_3_loss/Mean*
dtype0*
_output_shapes
: 
ђ
8training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/addAdd)loss/conv2d_3_loss/Mean/reduction_indices9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Size*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
: 

8training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/modFloorMod8training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/add9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Size*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
: 
Ћ
<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB **
_class 
loc:@loss/conv2d_3_loss/Mean
Ў
@training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range/startConst*
value	B : **
_class 
loc:@loss/conv2d_3_loss/Mean*
dtype0*
_output_shapes
: 
Ў
@training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range/deltaConst*
_output_shapes
: *
value	B :**
_class 
loc:@loss/conv2d_3_loss/Mean*
dtype0
ж
:training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/rangeRange@training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range/start9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Size@training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range/delta**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
:*

Tidx0
­
?training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Fill/valueConst*
value	B :**
_class 
loc:@loss/conv2d_3_loss/Mean*
dtype0*
_output_shapes
: 

9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/FillFill<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_1?training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Fill/value*
T0*

index_type0**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
: 

Btraining/Adam/gradients/loss/conv2d_3_loss/Mean_grad/DynamicStitchDynamicStitch:training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range8training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/mod:training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Fill*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*
N*
_output_shapes
:
Ќ
>training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum/yConst*
value	B :**
_class 
loc:@loss/conv2d_3_loss/Mean*
dtype0*
_output_shapes
: 

<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/MaximumMaximumBtraining/Adam/gradients/loss/conv2d_3_loss/Mean_grad/DynamicStitch>training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum/y*
_output_shapes
:*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean

=training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/floordivFloorDiv:training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
:
к
<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/ReshapeReshape>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/truedivBtraining/Adam/gradients/loss/conv2d_3_loss/Mean_grad/DynamicStitch*
T0*
Tshape0**
_class 
loc:@loss/conv2d_3_loss/Mean*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
б
9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/TileTile<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Reshape=training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/floordiv*

Tmultiples0*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
О
<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_2Shapeloss/conv2d_3_loss/Abs*
T0*
out_type0**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
:
П
<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_3Shapeloss/conv2d_3_loss/Mean*
_output_shapes
:*
T0*
out_type0**
_class 
loc:@loss/conv2d_3_loss/Mean
А
:training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/ConstConst*
valueB: **
_class 
loc:@loss/conv2d_3_loss/Mean*
dtype0*
_output_shapes
:
Ѕ
9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/ProdProd<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_2:training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Const**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
В
<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Const_1Const*
valueB: **
_class 
loc:@loss/conv2d_3_loss/Mean*
dtype0*
_output_shapes
:
Љ
;training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Prod_1Prod<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_3<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Const_1*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ў
@training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum_1/yConst*
value	B :**
_class 
loc:@loss/conv2d_3_loss/Mean*
dtype0*
_output_shapes
: 

>training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum_1Maximum;training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Prod_1@training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum_1/y*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
: 

?training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/floordiv_1FloorDiv9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Prod>training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum_1**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
: *
T0
ю
9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/CastCast?training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/floordiv_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0**
_class 
loc:@loss/conv2d_3_loss/Mean
Ѕ
<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/truedivRealDiv9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Tile9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Cast*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*1
_output_shapes
:џџџџџџџџџИа
П
8training/Adam/gradients/loss/conv2d_3_loss/Abs_grad/SignSignloss/conv2d_3_loss/sub*)
_class
loc:@loss/conv2d_3_loss/Abs*1
_output_shapes
:џџџџџџџџџИа*
T0

7training/Adam/gradients/loss/conv2d_3_loss/Abs_grad/mulMul<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/truediv8training/Adam/gradients/loss/conv2d_3_loss/Abs_grad/Sign*
T0*)
_class
loc:@loss/conv2d_3_loss/Abs*1
_output_shapes
:џџџџџџџџџИа
Д
9training/Adam/gradients/loss/conv2d_3_loss/sub_grad/ShapeShapeconv2d_3/BiasAdd*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/conv2d_3_loss/sub
Е
;training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape_1Shapeconv2d_3_target*
T0*
out_type0*)
_class
loc:@loss/conv2d_3_loss/sub*
_output_shapes
:
Т
Itraining/Adam/gradients/loss/conv2d_3_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape;training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape_1*)
_class
loc:@loss/conv2d_3_loss/sub*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
­
7training/Adam/gradients/loss/conv2d_3_loss/sub_grad/SumSum7training/Adam/gradients/loss/conv2d_3_loss/Abs_grad/mulItraining/Adam/gradients/loss/conv2d_3_loss/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*)
_class
loc:@loss/conv2d_3_loss/sub*
_output_shapes
:
Џ
;training/Adam/gradients/loss/conv2d_3_loss/sub_grad/ReshapeReshape7training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Sum9training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape*1
_output_shapes
:џџџџџџџџџИа*
T0*
Tshape0*)
_class
loc:@loss/conv2d_3_loss/sub
Б
9training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Sum_1Sum7training/Adam/gradients/loss/conv2d_3_loss/Abs_grad/mulKtraining/Adam/gradients/loss/conv2d_3_loss/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*)
_class
loc:@loss/conv2d_3_loss/sub*
_output_shapes
:
Ч
7training/Adam/gradients/loss/conv2d_3_loss/sub_grad/NegNeg9training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Sum_1*
_output_shapes
:*
T0*)
_class
loc:@loss/conv2d_3_loss/sub
Ь
=training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Reshape_1Reshape7training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Neg;training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape_1*
T0*
Tshape0*)
_class
loc:@loss/conv2d_3_loss/sub*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
ц
9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGrad;training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Reshape*#
_class
loc:@conv2d_3/BiasAdd*
data_formatNHWC*
_output_shapes
:*
T0
ь
8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNShapeN%up_sampling2d_1/ResizeNearestNeighborconv2d_3/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_3/convolution*
N* 
_output_shapes
::
М
Etraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNconv2d_3/kernel/read;training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Reshape*
T0*'
_class
loc:@conv2d_3/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа@*
	dilations

Ц
Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_1/ResizeNearestNeighbor:training/Adam/gradients/conv2d_3/convolution_grad/ShapeN:1;training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Reshape*'
_class
loc:@conv2d_3/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@*
	dilations
*
T0
ь
atraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"  (  *8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*
dtype0*
_output_shapes
:
Ў
\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradEtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputatraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
T0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*1
_output_shapes
:џџџџџџџџџЈ@*
align_corners( 
Ю
;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradSwitch\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGraddropout_2/cond/pred_id*
T0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@
о
training/Adam/gradients/SwitchSwitchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu
Ж
 training/Adam/gradients/IdentityIdentity training/Adam/gradients/Switch:1*1
_output_shapes
:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu
Ћ
training/Adam/gradients/Shape_1Shape training/Adam/gradients/Switch:1*
_output_shapes
:*
T0*
out_type0**
_class 
loc:@leaky_re_lu_2/LeakyRelu
З
#training/Adam/gradients/zeros/ConstConst!^training/Adam/gradients/Identity*
valueB
 *    **
_class 
loc:@leaky_re_lu_2/LeakyRelu*
dtype0*
_output_shapes
: 
х
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*1
_output_shapes
:џџџџџџџџџЈ@*
T0*

index_type0**
_class 
loc:@leaky_re_lu_2/LeakyRelu

>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros*3
_output_shapes!
:џџџџџџџџџЈ@: *
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N
Ъ
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ShapeShapedropout_2/cond/dropout/truediv*
_output_shapes
:*
T0*
out_type0*-
_class#
!loc:@dropout_2/cond/dropout/mul
Ъ
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1Shapedropout_2/cond/dropout/Floor*
out_type0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:*
T0
в
Mtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1dropout_2/cond/dropout/Floor*1
_output_shapes
:џџџџџџџџџЈ@*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
Н
;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
П
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape*
T0*
Tshape0*-
_class#
!loc:@dropout_2/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџЈ@

=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Muldropout_2/cond/dropout/truediv=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџЈ@
У
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
Х
Atraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*1
_output_shapes
:џџџџџџџџџЈ@*
T0*
Tshape0*-
_class#
!loc:@dropout_2/cond/dropout/mul
Ц
Atraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/ShapeShapedropout_2/cond/mul*
T0*
out_type0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
_output_shapes
:
Й
Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Shape_1Const*
valueB *1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
dtype0*
_output_shapes
: 
т
Qtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsAtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/ShapeCtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDivRealDiv?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshapedropout_2/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@
б
?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/SumSumCtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDivQtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv
Я
Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/ReshapeReshape?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/SumAtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Shape*
T0*
Tshape0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@
Щ
?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/NegNegdropout_2/cond/mul*1
_output_shapes
:џџџџџџџџџЈ@*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv

Etraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_1RealDiv?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Negdropout_2/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@
Ђ
Etraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_2RealDivEtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_1dropout_2/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџЈ@*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv
Н
?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/mulMul?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeEtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_2*1
_output_shapes
:џџџџџџџџџЈ@*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv
б
Atraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Sum_1Sum?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/mulStraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
К
Etraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Reshape_1ReshapeAtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Sum_1Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Shape_1*
T0*
Tshape0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
_output_shapes
: 
З
5training/Adam/gradients/dropout_2/cond/mul_grad/ShapeShapedropout_2/cond/mul/Switch:1*
T0*
out_type0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:
Ё
7training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1Const*
valueB *%
_class
loc:@dropout_2/cond/mul*
dtype0*
_output_shapes
: 
В
Etraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_2/cond/mul_grad/Shape7training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1*%
_class
loc:@dropout_2/cond/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ј
3training/Adam/gradients/dropout_2/cond/mul_grad/MulMulCtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Reshapedropout_2/cond/mul/y*
T0*%
_class
loc:@dropout_2/cond/mul*1
_output_shapes
:џџџџџџџџџЈ@

3training/Adam/gradients/dropout_2/cond/mul_grad/SumSum3training/Adam/gradients/dropout_2/cond/mul_grad/MulEtraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:

7training/Adam/gradients/dropout_2/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_2/cond/mul_grad/Sum5training/Adam/gradients/dropout_2/cond/mul_grad/Shape*1
_output_shapes
:џџџџџџџџџЈ@*
T0*
Tshape0*%
_class
loc:@dropout_2/cond/mul

5training/Adam/gradients/dropout_2/cond/mul_grad/Mul_1Muldropout_2/cond/mul/Switch:1Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Reshape*
T0*%
_class
loc:@dropout_2/cond/mul*1
_output_shapes
:џџџџџџџџџЈ@
Ѓ
5training/Adam/gradients/dropout_2/cond/mul_grad/Sum_1Sum5training/Adam/gradients/dropout_2/cond/mul_grad/Mul_1Gtraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 

9training/Adam/gradients/dropout_2/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_2/cond/mul_grad/Sum_17training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1*
T0*
Tshape0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
: 
р
 training/Adam/gradients/Switch_1Switchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id**
_class 
loc:@leaky_re_lu_2/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@*
T0
И
"training/Adam/gradients/Identity_1Identity training/Adam/gradients/Switch_1*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*1
_output_shapes
:џџџџџџџџџЈ@
Ћ
training/Adam/gradients/Shape_2Shape training/Adam/gradients/Switch_1*
T0*
out_type0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
_output_shapes
:
Л
%training/Adam/gradients/zeros_1/ConstConst#^training/Adam/gradients/Identity_1*
valueB
 *    **
_class 
loc:@leaky_re_lu_2/LeakyRelu*
dtype0*
_output_shapes
: 
щ
training/Adam/gradients/zeros_1Filltraining/Adam/gradients/Shape_2%training/Adam/gradients/zeros_1/Const*

index_type0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*1
_output_shapes
:џџџџџџџџџЈ@*
T0

@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_17training/Adam/gradients/dropout_2/cond/mul_grad/Reshape*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџЈ@: 

training/Adam/gradients/AddNAddN>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_grad*1
_output_shapes
:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N
ћ
Btraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGradLeakyReluGradtraining/Adam/gradients/AddNconv2d_2/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
alpha%>*1
_output_shapes
:џџџџџџџџџЈ@
э
9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_2/BiasAdd*
data_formatNHWC*
_output_shapes
:@
о
8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNShapeNmax_pooling2d_1/MaxPoolconv2d_2/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_2/convolution*
N* 
_output_shapes
::
У
Etraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNconv2d_2/kernel/readBtraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ 
П
Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_1/MaxPool:training/Adam/gradients/conv2d_2/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @*
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution
џ
@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGraddropout_1/cond/Mergemax_pooling2d_1/MaxPoolEtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа *
T0**
_class 
loc:@max_pooling2d_1/MaxPool*
strides
*
data_formatNHWC
Є
;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradSwitch@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGraddropout_1/cond/pred_id*
T0**
_class 
loc:@max_pooling2d_1/MaxPool*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа 
р
 training/Adam/gradients/Switch_2Switchleaky_re_lu_1/LeakyReludropout_1/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа 
К
"training/Adam/gradients/Identity_2Identity"training/Adam/gradients/Switch_2:1*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*1
_output_shapes
:џџџџџџџџџИа 
­
training/Adam/gradients/Shape_3Shape"training/Adam/gradients/Switch_2:1*
T0*
out_type0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
_output_shapes
:
Л
%training/Adam/gradients/zeros_2/ConstConst#^training/Adam/gradients/Identity_2*
valueB
 *    **
_class 
loc:@leaky_re_lu_1/LeakyRelu*
dtype0*
_output_shapes
: 
щ
training/Adam/gradients/zeros_2Filltraining/Adam/gradients/Shape_3%training/Adam/gradients/zeros_2/Const*1
_output_shapes
:џџџџџџџџџИа *
T0*

index_type0**
_class 
loc:@leaky_re_lu_1/LeakyRelu

>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_2*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџИа : 
Ъ
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ShapeShapedropout_1/cond/dropout/truediv*
T0*
out_type0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:
Ъ
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1Shapedropout_1/cond/dropout/Floor*
T0*
out_type0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:
в
Mtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*-
_class#
!loc:@dropout_1/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1dropout_1/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџИа 
Н
;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
П
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape*
T0*
Tshape0*-
_class#
!loc:@dropout_1/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџИа 

=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Muldropout_1/cond/dropout/truediv=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџИа 
У
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Х
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*1
_output_shapes
:џџџџџџџџџИа *
T0*
Tshape0*-
_class#
!loc:@dropout_1/cond/dropout/mul
Ц
Atraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/ShapeShapedropout_1/cond/mul*
T0*
out_type0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
:
Й
Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1Const*
_output_shapes
: *
valueB *1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
dtype0
т
Qtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsAtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/ShapeCtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDivRealDiv?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshapedropout_1/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа 
б
?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/SumSumCtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDivQtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgs*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
Я
Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/ReshapeReshape?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/SumAtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Shape*1
_output_shapes
:џџџџџџџџџИа *
T0*
Tshape0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
Щ
?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/NegNegdropout_1/cond/mul*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа *
T0

Etraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1RealDiv?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Negdropout_1/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа 
Ђ
Etraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2RealDivEtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1dropout_1/cond/dropout/sub*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа *
T0
Н
?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/mulMul?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeEtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2*1
_output_shapes
:џџџџџџџџџИа *
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
б
Atraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1Sum?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/mulStraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
:
К
Etraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Reshape_1ReshapeAtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1*
T0*
Tshape0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
: 
З
5training/Adam/gradients/dropout_1/cond/mul_grad/ShapeShapedropout_1/cond/mul/Switch:1*
_output_shapes
:*
T0*
out_type0*%
_class
loc:@dropout_1/cond/mul
Ё
7training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *%
_class
loc:@dropout_1/cond/mul
В
Etraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_1/cond/mul_grad/Shape7training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@dropout_1/cond/mul
ј
3training/Adam/gradients/dropout_1/cond/mul_grad/MulMulCtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Reshapedropout_1/cond/mul/y*
T0*%
_class
loc:@dropout_1/cond/mul*1
_output_shapes
:џџџџџџџџџИа 

3training/Adam/gradients/dropout_1/cond/mul_grad/SumSum3training/Adam/gradients/dropout_1/cond/mul_grad/MulEtraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs*
T0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 

7training/Adam/gradients/dropout_1/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_1/cond/mul_grad/Sum5training/Adam/gradients/dropout_1/cond/mul_grad/Shape*
T0*
Tshape0*%
_class
loc:@dropout_1/cond/mul*1
_output_shapes
:џџџџџџџџџИа 

5training/Adam/gradients/dropout_1/cond/mul_grad/Mul_1Muldropout_1/cond/mul/Switch:1Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Reshape*%
_class
loc:@dropout_1/cond/mul*1
_output_shapes
:џџџџџџџџџИа *
T0
Ѓ
5training/Adam/gradients/dropout_1/cond/mul_grad/Sum_1Sum5training/Adam/gradients/dropout_1/cond/mul_grad/Mul_1Gtraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:

9training/Adam/gradients/dropout_1/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_1/cond/mul_grad/Sum_17training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1*
Tshape0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
: *
T0
р
 training/Adam/gradients/Switch_3Switchleaky_re_lu_1/LeakyReludropout_1/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа 
И
"training/Adam/gradients/Identity_3Identity training/Adam/gradients/Switch_3*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*1
_output_shapes
:џџџџџџџџџИа 
Ћ
training/Adam/gradients/Shape_4Shape training/Adam/gradients/Switch_3*
T0*
out_type0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
_output_shapes
:
Л
%training/Adam/gradients/zeros_3/ConstConst#^training/Adam/gradients/Identity_3*
valueB
 *    **
_class 
loc:@leaky_re_lu_1/LeakyRelu*
dtype0*
_output_shapes
: 
щ
training/Adam/gradients/zeros_3Filltraining/Adam/gradients/Shape_4%training/Adam/gradients/zeros_3/Const*

index_type0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*1
_output_shapes
:џџџџџџџџџИа *
T0

@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_37training/Adam/gradients/dropout_1/cond/mul_grad/Reshape*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџИа : 

training/Adam/gradients/AddN_1AddN>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_grad**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
N*1
_output_shapes
:џџџџџџџџџИа *
T0
§
Btraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGradLeakyReluGradtraining/Adam/gradients/AddN_1conv2d_1/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
alpha%>*1
_output_shapes
:џџџџџџџџџИа 
э
9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_1/BiasAdd*
data_formatNHWC*
_output_shapes
: 
е
8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNShapeNconv2d_1_inputconv2d_1/kernel/read* 
_output_shapes
::*
T0*
out_type0*'
_class
loc:@conv2d_1/convolution*
N
У
Etraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNconv2d_1/kernel/readBtraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа*
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
Ж
Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_1_input:training/Adam/gradients/conv2d_1/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*'
_class
loc:@conv2d_1/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0
_
training/Adam/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Ќ
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
use_locking( *
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
p
training/Adam/CastCastAdam/iterations/read*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
X
training/Adam/add/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
T0*
_output_shapes
: 
^
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
T0*
_output_shapes
: 
X
training/Adam/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_1Const*
valueB
 *  *
dtype0*
_output_shapes
: 
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
T0*
_output_shapes
: 

training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
T0*
_output_shapes
: 
`
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add*
_output_shapes
: *
T0
Z
training/Adam/sub_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
_output_shapes
: *
T0
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
_output_shapes
: *
T0
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
T0*
_output_shapes
: 
x
training/Adam/zerosConst*
dtype0*&
_output_shapes
: *%
valueB *    

training/Adam/Variable
VariableV2*&
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
й
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*&
_output_shapes
: *
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(

training/Adam/Variable/readIdentitytraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*&
_output_shapes
: *
T0
b
training/Adam/zeros_1Const*
valueB *    *
dtype0*
_output_shapes
: 

training/Adam/Variable_1
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
е
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
_output_shapes
: *
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
: 
~
%training/Adam/zeros_2/shape_as_tensorConst*%
valueB"          @   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Є
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*
T0*

index_type0*&
_output_shapes
: @

training/Adam/Variable_2
VariableV2*
dtype0*&
_output_shapes
: @*
	container *
shape: @*
shared_name 
с
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*&
_output_shapes
: @
Ё
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*&
_output_shapes
: @*
T0*+
_class!
loc:@training/Adam/Variable_2
b
training/Adam/zeros_3Const*
_output_shapes
:@*
valueB@*    *
dtype0

training/Adam/Variable_3
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
е
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0

training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
T0*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
:@
~
%training/Adam/zeros_4/shape_as_tensorConst*%
valueB"      @      *
dtype0*
_output_shapes
:
`
training/Adam/zeros_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Є
training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*
T0*

index_type0*&
_output_shapes
:@

training/Adam/Variable_4
VariableV2*
dtype0*&
_output_shapes
:@*
	container *
shape:@*
shared_name 
с
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4
Ё
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4*&
_output_shapes
:@
b
training/Adam/zeros_5Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_5
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
е
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
:
z
training/Adam/zeros_6Const*%
valueB *    *
dtype0*&
_output_shapes
: 

training/Adam/Variable_6
VariableV2*
shape: *
shared_name *
dtype0*&
_output_shapes
: *
	container 
с
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0
Ё
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
T0*+
_class!
loc:@training/Adam/Variable_6*&
_output_shapes
: 
b
training/Adam/zeros_7Const*
valueB *    *
dtype0*
_output_shapes
: 

training/Adam/Variable_7
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
е
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
: 
~
%training/Adam/zeros_8/shape_as_tensorConst*%
valueB"          @   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_8/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Є
training/Adam/zeros_8Fill%training/Adam/zeros_8/shape_as_tensortraining/Adam/zeros_8/Const*

index_type0*&
_output_shapes
: @*
T0

training/Adam/Variable_8
VariableV2*
dtype0*&
_output_shapes
: @*
	container *
shape: @*
shared_name 
с
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*&
_output_shapes
: @
Ё
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
T0*+
_class!
loc:@training/Adam/Variable_8*&
_output_shapes
: @
b
training/Adam/zeros_9Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_9
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
е
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9

training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
:@

&training/Adam/zeros_10/shape_as_tensorConst*%
valueB"      @      *
dtype0*
_output_shapes
:
a
training/Adam/zeros_10/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ї
training/Adam/zeros_10Fill&training/Adam/zeros_10/shape_as_tensortraining/Adam/zeros_10/Const*
T0*

index_type0*&
_output_shapes
:@

training/Adam/Variable_10
VariableV2*
shared_name *
dtype0*&
_output_shapes
:@*
	container *
shape:@
х
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10
Є
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*&
_output_shapes
:@*
T0
c
training/Adam/zeros_11Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_11
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
:
p
&training/Adam/zeros_12/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_12/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_12Fill&training/Adam/zeros_12/shape_as_tensortraining/Adam/zeros_12/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_12
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes
:

training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes
:
p
&training/Adam/zeros_13/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_13/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_13Fill&training/Adam/zeros_13/shape_as_tensortraining/Adam/zeros_13/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_13
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:

training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
T0*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
:
p
&training/Adam/zeros_14/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_14/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_14
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes
:

training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes
:
p
&training/Adam/zeros_15/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_15/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_15Fill&training/Adam/zeros_15/shape_as_tensortraining/Adam/zeros_15/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_15
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
:
p
&training/Adam/zeros_16/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_16
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
_output_shapes
:*
T0
p
&training/Adam/zeros_17/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_17/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_17Fill&training/Adam/zeros_17/shape_as_tensortraining/Adam/zeros_17/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_17
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
_output_shapes
:
z
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*&
_output_shapes
: *
T0
Z
training/Adam/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ј
training/Adam/mul_2Multraining/Adam/sub_2Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
u
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*&
_output_shapes
: 
|
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_6/read*
T0*&
_output_shapes
: 
Z
training/Adam/sub_3/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/SquareSquareFtraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
v
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*&
_output_shapes
: 
u
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*&
_output_shapes
: 
s
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*&
_output_shapes
: *
T0
Z
training/Adam/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_3Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*
T0*&
_output_shapes
: 

training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
T0*&
_output_shapes
: 
l
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*&
_output_shapes
: 
Z
training/Adam/add_3/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
x
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*&
_output_shapes
: 
}
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*&
_output_shapes
: *
T0
z
training/Adam/sub_4Subconv2d_1/kernel/readtraining/Adam/truediv_1*&
_output_shapes
: *
T0
а
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*&
_output_shapes
: *
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(
ж
training/Adam/Assign_1Assigntraining/Adam/Variable_6training/Adam/add_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*&
_output_shapes
: 
Ф
training/Adam/Assign_2Assignconv2d_1/kerneltraining/Adam/sub_4*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
: 
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
_output_shapes
: *
T0
Z
training/Adam/sub_5/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_7Multraining/Adam/sub_59training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
: 
p
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_7/read*
T0*
_output_shapes
: 
Z
training/Adam/sub_6/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_1Square9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes
: 
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
: 
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
_output_shapes
: *
T0
Z
training/Adam/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_5Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
T0*
_output_shapes
: 

training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
T0*
_output_shapes
: 
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
: 
Z
training/Adam/add_6/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
_output_shapes
: *
T0
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
_output_shapes
: *
T0
l
training/Adam/sub_7Subconv2d_1/bias/readtraining/Adam/truediv_2*
_output_shapes
: *
T0
Ъ
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ъ
training/Adam/Assign_4Assigntraining/Adam/Variable_7training/Adam/add_5*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes
: 
Д
training/Adam/Assign_5Assignconv2d_1/biastraining/Adam/sub_7*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(
}
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0*&
_output_shapes
: @
Z
training/Adam/sub_8/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
T0*
_output_shapes
: 
Љ
training/Adam/mul_12Multraining/Adam/sub_8Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
w
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*&
_output_shapes
: @
}
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_8/read*&
_output_shapes
: @*
T0
Z
training/Adam/sub_9/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_2SquareFtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
y
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*&
_output_shapes
: @*
T0
w
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*&
_output_shapes
: @*
T0
t
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*&
_output_shapes
: @*
T0
Z
training/Adam/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_7Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*
T0*&
_output_shapes
: @

training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
T0*&
_output_shapes
: @
l
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*&
_output_shapes
: @*
T0
Z
training/Adam/add_9/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
x
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0*&
_output_shapes
: @
~
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*&
_output_shapes
: @*
T0
{
training/Adam/sub_10Subconv2d_2/kernel/readtraining/Adam/truediv_3*
T0*&
_output_shapes
: @
ж
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*&
_output_shapes
: @
ж
training/Adam/Assign_7Assigntraining/Adam/Variable_8training/Adam/add_8*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*&
_output_shapes
: @
Х
training/Adam/Assign_8Assignconv2d_2/kerneltraining/Adam/sub_10*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
: @
q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
T0*
_output_shapes
:@
[
training/Adam/sub_11/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_17Multraining/Adam/sub_119training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
:@
q
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_9/read*
T0*
_output_shapes
:@
[
training/Adam/sub_12/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_3Square9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
_output_shapes
:@*
T0
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
_output_shapes
:@*
T0
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
_output_shapes
:@*
T0
Z
training/Adam/Const_8Const*
_output_shapes
: *
valueB
 *    *
dtype0
Z
training/Adam/Const_9Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
T0*
_output_shapes
:@

training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
_output_shapes
:@*
T0
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes
:@
[
training/Adam/add_12/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
_output_shapes
:@*
T0
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
_output_shapes
:@*
T0
m
training/Adam/sub_13Subconv2d_2/bias/readtraining/Adam/truediv_4*
_output_shapes
:@*
T0
Ы
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:@
Ь
training/Adam/Assign_10Assigntraining/Adam/Variable_9training/Adam/add_11*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:@
Ж
training/Adam/Assign_11Assignconv2d_2/biastraining/Adam/sub_13*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@
}
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*&
_output_shapes
:@*
T0
[
training/Adam/sub_14/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
T0*
_output_shapes
: 
Њ
training/Adam/mul_22Multraining/Adam/sub_14Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
x
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*&
_output_shapes
:@*
T0
~
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_10/read*&
_output_shapes
:@*
T0
[
training/Adam/sub_15/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_15Subtraining/Adam/sub_15/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_4SquareFtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
z
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*&
_output_shapes
:@*
T0
x
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*&
_output_shapes
:@*
T0
u
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*&
_output_shapes
:@*
T0
[
training/Adam/Const_10Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_11Const*
dtype0*
_output_shapes
: *
valueB
 *  

%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*
T0*&
_output_shapes
:@

training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
T0*&
_output_shapes
:@
l
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*&
_output_shapes
:@
[
training/Adam/add_15/yConst*
dtype0*
_output_shapes
: *
valueB
 *Пж3
z
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*&
_output_shapes
:@

training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*&
_output_shapes
:@*
T0
{
training/Adam/sub_16Subconv2d_3/kernel/readtraining/Adam/truediv_5*
T0*&
_output_shapes
:@
и
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4
к
training/Adam/Assign_13Assigntraining/Adam/Variable_10training/Adam/add_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*&
_output_shapes
:@
Ц
training/Adam/Assign_14Assignconv2d_3/kerneltraining/Adam/sub_16*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(*&
_output_shapes
:@
q
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
_output_shapes
:*
T0
[
training/Adam/sub_17/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_27Multraining/Adam/sub_179training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
T0*
_output_shapes
:
r
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_11/read*
T0*
_output_shapes
:
[
training/Adam/sub_18/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_5Square9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
_output_shapes
:*
T0
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
_output_shapes
:*
T0
i
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes
:
[
training/Adam/Const_12Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_13Const*
dtype0*
_output_shapes
: *
valueB
 *  

%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
T0*
_output_shapes
:

training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
_output_shapes
:*
T0
`
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
_output_shapes
:*
T0
[
training/Adam/add_18/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
n
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
_output_shapes
:*
T0
s
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0*
_output_shapes
:
m
training/Adam/sub_19Subconv2d_3/bias/readtraining/Adam/truediv_6*
_output_shapes
:*
T0
Ь
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:
Ю
training/Adam/Assign_16Assigntraining/Adam/Variable_11training/Adam/add_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:
Ж
training/Adam/Assign_17Assignconv2d_3/biastraining/Adam/sub_19* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/group_depsNoOp	^loss/mul^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_2^training/Adam/Assign_3^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9


group_depsNoOp	^loss/mul

IsVariableInitializedIsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_2IsVariableInitializedconv2d_2/kernel*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel*
dtype0

IsVariableInitialized_3IsVariableInitializedconv2d_2/bias*
dtype0*
_output_shapes
: * 
_class
loc:@conv2d_2/bias

IsVariableInitialized_4IsVariableInitializedconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializedconv2d_3/bias*
dtype0*
_output_shapes
: * 
_class
loc:@conv2d_3/bias

IsVariableInitialized_6IsVariableInitializedAdam/iterations*
dtype0	*
_output_shapes
: *"
_class
loc:@Adam/iterations
z
IsVariableInitialized_7IsVariableInitializedAdam/lr*
_output_shapes
: *
_class
loc:@Adam/lr*
dtype0

IsVariableInitialized_8IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_9IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 

IsVariableInitialized_11IsVariableInitializedtraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
dtype0*
_output_shapes
: 

IsVariableInitialized_12IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_13IsVariableInitializedtraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_14IsVariableInitializedtraining/Adam/Variable_3*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_3

IsVariableInitialized_15IsVariableInitializedtraining/Adam/Variable_4*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_4

IsVariableInitialized_16IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
: 

IsVariableInitialized_17IsVariableInitializedtraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes
: 

IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable_7*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_7*
dtype0

IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes
: 

IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0*
_output_shapes
: 

IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
: 

IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
: 

IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
dtype0*
_output_shapes
: 

IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes
: 

IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_15*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_15

IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_16*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_16*
dtype0

IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
dtype0*
_output_shapes
: 
ю
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^conv2d_3/bias/Assign^conv2d_3/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign ^training/Adam/Variable_2/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign"кС#Ѓ§     mC36	`:#VК зAJћ
у*С*
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
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
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
6
Pow
x"T
y"T
z"T"
Ttype:

2	
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
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
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
shared_namestring *1.13.12v1.13.1-0-g6612da8951Щ

conv2d_1_inputPlaceholder*
dtype0*1
_output_shapes
:џџџџџџџџџИа*&
shape:џџџџџџџџџИа
v
conv2d_1/random_uniform/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
`
conv2d_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *OSО
`
conv2d_1/random_uniform/maxConst*
valueB
 *OS>*
dtype0*
_output_shapes
: 
В
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
seedБџх)*
T0*
dtype0*
seed2ГЄ*&
_output_shapes
: 
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
_output_shapes
: *
T0

conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*&
_output_shapes
: *
T0

conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*
T0*&
_output_shapes
: 

conv2d_1/kernel
VariableV2*
shared_name *
dtype0*
	container *&
_output_shapes
: *
shape: 
Ш
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
: 

conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: 
[
conv2d_1/ConstConst*
valueB *    *
dtype0*
_output_shapes
: 
y
conv2d_1/bias
VariableV2*
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 
­
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
: 
t
conv2d_1/bias/readIdentityconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
_output_shapes
: *
T0
s
"conv2d_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ю
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*1
_output_shapes
:џџџџџџџџџИа *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа 

leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_1/BiasAdd*
alpha%>*1
_output_shapes
:џџџџџџџџџИа *
T0
f
$dropout_1/keras_learning_phase/inputConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
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
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
_output_shapes
: *
T0

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
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 

dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*
T0*1
_output_shapes
:џџџџџџџџџИа 
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
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
_output_shapes
:*
T0*
out_type0
{
dropout_1/cond/dropout/sub/xConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
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
dtype0*
seed2кл*1
_output_shapes
:џџџџџџџџџИа *
seedБџх)
Ї
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ь
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*1
_output_shapes
:џџџџџџџџџИа 
О
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*1
_output_shapes
:џџџџџџџџџИа *
T0
 
dropout_1/cond/dropout/addAdddropout_1/cond/dropout/sub%dropout_1/cond/dropout/random_uniform*1
_output_shapes
:џџџџџџџџџИа *
T0
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
dropout_1/cond/Switch_1Switchleaky_re_lu_1/LeakyReludropout_1/cond/pred_id**
_class 
loc:@leaky_re_lu_1/LeakyRelu*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа *
T0
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
conv2d_2/random_uniform/shapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
`
conv2d_2/random_uniform/minConst*
valueB
 *ЋЊЊН*
dtype0*
_output_shapes
: 
`
conv2d_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ЋЊЊ=
В
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
dtype0*
seed2ђшр*&
_output_shapes
: @*
seedБџх)*
T0
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
T0*
_output_shapes
: 

conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*&
_output_shapes
: @*
T0

conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*
T0*&
_output_shapes
: @

conv2d_2/kernel
VariableV2*
shared_name *
dtype0*
	container *&
_output_shapes
: @*
shape: @
Ш
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
: @*
use_locking(

conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
: @
[
conv2d_2/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_2/bias
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:@*
shape:@
­
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias
t
conv2d_2/bias/readIdentityconv2d_2/bias*
_output_shapes
:@*
T0* 
_class
loc:@conv2d_2/bias
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ї
conv2d_2/convolutionConv2Dmax_pooling2d_1/MaxPoolconv2d_2/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ@

conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*1
_output_shapes
:џџџџџџџџџЈ@*
T0*
data_formatNHWC

leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_2/BiasAdd*
T0*
alpha%>*1
_output_shapes
:џџџџџџџџџЈ@

dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes
: : *
T0

]
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
_output_shapes
: *
T0

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
dropout_2/cond/mul/SwitchSwitchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id**
_class 
loc:@leaky_re_lu_2/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@*
T0
z
dropout_2/cond/dropout/rateConst^dropout_2/cond/switch_t*
valueB
 *ЭЬЬ=*
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
dropout_2/cond/dropout/sub/xConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
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
dtype0*
seed2Пј*1
_output_shapes
:џџџџџџџџџЈ@*
seedБџх)
Ї
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
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
dropout_2/cond/dropout/addAdddropout_2/cond/dropout/sub%dropout_2/cond/dropout/random_uniform*1
_output_shapes
:џџџџџџџџџЈ@*
T0
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
dropout_2/cond/Switch_1Switchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@

dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
N*3
_output_shapes!
:џџџџџџџџџЈ@: *
T0
i
up_sampling2d_1/ShapeShapedropout_2/cond/Merge*
out_type0*
_output_shapes
:*
T0
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
up_sampling2d_1/mulMulup_sampling2d_1/strided_sliceup_sampling2d_1/Const*
T0*
_output_shapes
:
К
%up_sampling2d_1/ResizeNearestNeighborResizeNearestNeighbordropout_2/cond/Mergeup_sampling2d_1/mul*
align_corners( *
T0*1
_output_shapes
:џџџџџџџџџИа@
v
conv2d_3/random_uniform/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
`
conv2d_3/random_uniform/minConst*
valueB
 *8JЬН*
dtype0*
_output_shapes
: 
`
conv2d_3/random_uniform/maxConst*
valueB
 *8JЬ=*
dtype0*
_output_shapes
: 
В
%conv2d_3/random_uniform/RandomUniformRandomUniformconv2d_3/random_uniform/shape*
seedБџх)*
T0*
dtype0*
seed2§щ*&
_output_shapes
:@
}
conv2d_3/random_uniform/subSubconv2d_3/random_uniform/maxconv2d_3/random_uniform/min*
T0*
_output_shapes
: 

conv2d_3/random_uniform/mulMul%conv2d_3/random_uniform/RandomUniformconv2d_3/random_uniform/sub*
T0*&
_output_shapes
:@

conv2d_3/random_uniformAddconv2d_3/random_uniform/mulconv2d_3/random_uniform/min*
T0*&
_output_shapes
:@

conv2d_3/kernel
VariableV2*
shape:@*
shared_name *
dtype0*
	container *&
_output_shapes
:@
Ш
conv2d_3/kernel/AssignAssignconv2d_3/kernelconv2d_3/random_uniform*&
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(

conv2d_3/kernel/readIdentityconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:@*
T0
[
conv2d_3/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv2d_3/bias
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
­
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/Const* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
t
conv2d_3/bias/readIdentityconv2d_3/bias*
T0* 
_class
loc:@conv2d_3/bias*
_output_shapes
:
s
"conv2d_3/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0

conv2d_3/convolutionConv2D%up_sampling2d_1/ResizeNearestNeighborconv2d_3/kernel/read*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv2d_3/BiasAddBiasAddconv2d_3/convolutionconv2d_3/bias/read*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа*
T0
_
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
shared_name *
dtype0	*
	container *
_output_shapes
: *
shape: 
О
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
_output_shapes
: *
use_locking(*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(
v
Adam/iterations/readIdentityAdam/iterations*
_output_shapes
: *
T0	*"
_class
loc:@Adam/iterations
Z
Adam/lr/initial_valueConst*
valueB
 *Зб8*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 

Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/lr
^
Adam/lr/readIdentityAdam/lr*
T0*
_class
loc:@Adam/lr*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
Adam/beta_1
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
Ў
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: 
j
Adam/beta_1/readIdentityAdam/beta_1*
_output_shapes
: *
T0*
_class
loc:@Adam/beta_1
^
Adam/beta_2/initial_valueConst*
valueB
 *wО?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
Ў
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_2*
validate_shape(
j
Adam/beta_2/readIdentityAdam/beta_2*
_class
loc:@Adam/beta_2*
_output_shapes
: *
T0
]
Adam/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

Adam/decay
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
Њ
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
T0*
_class
loc:@Adam/decay*
validate_shape(*
_output_shapes
: *
use_locking(
g
Adam/decay/readIdentity
Adam/decay*
_class
loc:@Adam/decay*
_output_shapes
: *
T0
И
conv2d_3_targetPlaceholder*?
shape6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
r
conv2d_3_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
|
loss/conv2d_3_loss/subSubconv2d_3/BiasAddconv2d_3_target*
T0*1
_output_shapes
:џџџџџџџџџИа
q
loss/conv2d_3_loss/AbsAbsloss/conv2d_3_loss/sub*1
_output_shapes
:џџџџџџџџџИа*
T0
t
)loss/conv2d_3_loss/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
З
loss/conv2d_3_loss/MeanMeanloss/conv2d_3_loss/Abs)loss/conv2d_3_loss/Mean/reduction_indices*
T0*-
_output_shapes
:џџџџџџџџџИа*

Tidx0*
	keep_dims( 
|
+loss/conv2d_3_loss/Mean_1/reduction_indicesConst*
_output_shapes
:*
valueB"      *
dtype0
В
loss/conv2d_3_loss/Mean_1Meanloss/conv2d_3_loss/Mean+loss/conv2d_3_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( 

loss/conv2d_3_loss/mulMulloss/conv2d_3_loss/Mean_1conv2d_3_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
b
loss/conv2d_3_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/conv2d_3_loss/NotEqualNotEqualconv2d_3_sample_weightsloss/conv2d_3_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ

loss/conv2d_3_loss/CastCastloss/conv2d_3_loss/NotEqual*
Truncate( *

DstT0*#
_output_shapes
:џџџџџџџџџ*

SrcT0

b
loss/conv2d_3_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/conv2d_3_loss/Mean_2Meanloss/conv2d_3_loss/Castloss/conv2d_3_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 

loss/conv2d_3_loss/truedivRealDivloss/conv2d_3_loss/mulloss/conv2d_3_loss/Mean_2*
T0*#
_output_shapes
:џџџџџџџџџ
d
loss/conv2d_3_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/conv2d_3_loss/Mean_3Meanloss/conv2d_3_loss/truedivloss/conv2d_3_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
W
loss/mulMul
loss/mul/xloss/conv2d_3_loss/Mean_3*
T0*
_output_shapes
: 
}
training/Adam/gradients/ShapeConst*
_class
loc:@loss/mul*
valueB *
dtype0*
_output_shapes
: 

!training/Adam/gradients/grad_ys_0Const*
_class
loc:@loss/mul*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ж
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*
_class
loc:@loss/mul*

index_type0*
_output_shapes
: 
Ї
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/conv2d_3_loss/Mean_3*
_output_shapes
: *
T0*
_class
loc:@loss/mul

+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
_class
loc:@loss/mul*
_output_shapes
: *
T0
М
Dtraining/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Reshape/shapeConst*
_output_shapes
:*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
valueB:*
dtype0

>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Dtraining/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Reshape/shape*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
Tshape0*
_output_shapes
:
Ф
<training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/ShapeShapeloss/conv2d_3_loss/truediv*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
out_type0*
_output_shapes
:
Џ
;training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/TileTile>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Reshape<training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape*

Tmultiples0*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Ц
>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape_1Shapeloss/conv2d_3_loss/truediv*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
out_type0*
_output_shapes
:
Џ
>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape_2Const*
_output_shapes
: *,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
valueB *
dtype0
Д
<training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/ConstConst*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
valueB: *
dtype0*
_output_shapes
:
­
;training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/ProdProd>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape_1<training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Const*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
_output_shapes
: *
	keep_dims( *

Tidx0
Ж
>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Const_1Const*
dtype0*
_output_shapes
:*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
valueB: 
Б
=training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Prod_1Prod>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape_2>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3
А
@training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Maximum/yConst*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
value	B :*
dtype0*
_output_shapes
: 

>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/MaximumMaximum=training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Prod_1@training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Maximum/y*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
_output_shapes
: 

?training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/floordivFloorDiv;training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Prod>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Maximum*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
_output_shapes
: 
ђ
;training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/CastCast?training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/floordiv*

SrcT0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*
Truncate( *

DstT0*
_output_shapes
: 

>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/truedivRealDiv;training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Tile;training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Cast*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Т
=training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/ShapeShapeloss/conv2d_3_loss/mul*
_output_shapes
:*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*
out_type0
Б
?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
: *-
_class#
!loc:@loss/conv2d_3_loss/truediv*
valueB 
в
Mtraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape_1*-
_class#
!loc:@loss/conv2d_3_loss/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDivRealDiv>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/truedivloss/conv2d_3_loss/Mean_2*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ
С
;training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/SumSum?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDivMtraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*
_output_shapes
:
Б
?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/ReshapeReshape;training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Sum=training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*
Tshape0*#
_output_shapes
:џџџџџџџџџ
З
;training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/NegNegloss/conv2d_3_loss/mul*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Atraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDiv_1RealDiv;training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Negloss/conv2d_3_loss/Mean_2*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Atraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDiv_2RealDivAtraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDiv_1loss/conv2d_3_loss/Mean_2*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ
Ђ
;training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/mulMul>training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/truedivAtraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDiv_2*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ
С
=training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Sum_1Sum;training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/mulOtraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*
_output_shapes
:
Њ
Atraining/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Reshape_1Reshape=training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Sum_1?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape_1*
_output_shapes
: *
T0*-
_class#
!loc:@loss/conv2d_3_loss/truediv*
Tshape0
Н
9training/Adam/gradients/loss/conv2d_3_loss/mul_grad/ShapeShapeloss/conv2d_3_loss/Mean_1*
_output_shapes
:*
T0*)
_class
loc:@loss/conv2d_3_loss/mul*
out_type0
Н
;training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape_1Shapeconv2d_3_sample_weights*
T0*)
_class
loc:@loss/conv2d_3_loss/mul*
out_type0*
_output_shapes
:
Т
Itraining/Adam/gradients/loss/conv2d_3_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape;training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape_1*)
_class
loc:@loss/conv2d_3_loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ё
7training/Adam/gradients/loss/conv2d_3_loss/mul_grad/MulMul?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Reshapeconv2d_3_sample_weights*
T0*)
_class
loc:@loss/conv2d_3_loss/mul*#
_output_shapes
:џџџџџџџџџ
­
7training/Adam/gradients/loss/conv2d_3_loss/mul_grad/SumSum7training/Adam/gradients/loss/conv2d_3_loss/mul_grad/MulItraining/Adam/gradients/loss/conv2d_3_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/conv2d_3_loss/mul*
_output_shapes
:
Ё
;training/Adam/gradients/loss/conv2d_3_loss/mul_grad/ReshapeReshape7training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Sum9training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape*
T0*)
_class
loc:@loss/conv2d_3_loss/mul*
Tshape0*#
_output_shapes
:џџџџџџџџџ
ѕ
9training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Mul_1Mulloss/conv2d_3_loss/Mean_1?training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Reshape*#
_output_shapes
:џџџџџџџџџ*
T0*)
_class
loc:@loss/conv2d_3_loss/mul
Г
9training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Sum_1Sum9training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Mul_1Ktraining/Adam/gradients/loss/conv2d_3_loss/mul_grad/BroadcastGradientArgs:1*
T0*)
_class
loc:@loss/conv2d_3_loss/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Ї
=training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Reshape_1Reshape9training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Sum_1;training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape_1*
T0*)
_class
loc:@loss/conv2d_3_loss/mul*
Tshape0*#
_output_shapes
:џџџџџџџџџ
С
<training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/ShapeShapeloss/conv2d_3_loss/Mean*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
out_type0*
_output_shapes
:
Ћ
;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/SizeConst*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
ў
:training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/addAdd+loss/conv2d_3_loss/Mean_1/reduction_indices;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Size*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
:

:training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/modFloorMod:training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/add;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Size*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
:
Ж
>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_1Const*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
valueB:*
dtype0*
_output_shapes
:
В
Btraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range/startConst*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
value	B : *
dtype0*
_output_shapes
: 
В
Btraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range/deltaConst*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
р
<training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/rangeRangeBtraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range/start;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/SizeBtraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range/delta*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
:*

Tidx0
Б
Atraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Fill/valueConst*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
Ћ
;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/FillFill>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_1Atraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Fill/value*
_output_shapes
:*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*

index_type0
І
Dtraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/DynamicStitchDynamicStitch<training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range:training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/mod<training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Fill*
_output_shapes
:*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
N
А
@training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum/yConst*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
Є
>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/MaximumMaximumDtraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/DynamicStitch@training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum/y*
_output_shapes
:*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1

?training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/floordivFloorDiv<training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
:
а
>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/ReshapeReshape;training/Adam/gradients/loss/conv2d_3_loss/mul_grad/ReshapeDtraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/DynamicStitch*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
Tshape0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ь
;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/TileTile>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Reshape?training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/floordiv*

Tmultiples0*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
У
>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_2Shapeloss/conv2d_3_loss/Mean*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
out_type0*
_output_shapes
:*
T0
Х
>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_3Shapeloss/conv2d_3_loss/Mean_1*
_output_shapes
:*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
out_type0
Д
<training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/ConstConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
valueB: 
­
;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/ProdProd>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_2<training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Const*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
: *
	keep_dims( *

Tidx0
Ж
>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Const_1Const*
_output_shapes
:*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
valueB: *
dtype0
Б
=training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Prod_1Prod>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_3>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Const_1*
	keep_dims( *

Tidx0*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
: 
В
Btraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum_1/yConst*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 

@training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum_1Maximum=training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Prod_1Btraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum_1/y*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
: 

Atraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/floordiv_1FloorDiv;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Prod@training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum_1*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
_output_shapes
: 
є
;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/CastCastAtraining/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/floordiv_1*

DstT0*
_output_shapes
: *

SrcT0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*
Truncate( 
Љ
>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/truedivRealDiv;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Tile;training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Cast*
T0*,
_class"
 loc:@loss/conv2d_3_loss/Mean_1*-
_output_shapes
:џџџџџџџџџИа
М
:training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/ShapeShapeloss/conv2d_3_loss/Abs*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*
out_type0*
_output_shapes
:
Ї
9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/SizeConst**
_class 
loc:@loss/conv2d_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
ђ
8training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/addAdd)loss/conv2d_3_loss/Mean/reduction_indices9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Size*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
: 

8training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/modFloorMod8training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/add9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Size**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
: *
T0
Ћ
<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_1Const**
_class 
loc:@loss/conv2d_3_loss/Mean*
valueB *
dtype0*
_output_shapes
: 
Ў
@training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range/startConst**
_class 
loc:@loss/conv2d_3_loss/Mean*
value	B : *
dtype0*
_output_shapes
: 
Ў
@training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range/deltaConst**
_class 
loc:@loss/conv2d_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
ж
:training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/rangeRange@training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range/start9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Size@training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range/delta**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
:*

Tidx0
­
?training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Fill/valueConst**
_class 
loc:@loss/conv2d_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/FillFill<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_1?training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Fill/value*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*

index_type0*
_output_shapes
: 

Btraining/Adam/gradients/loss/conv2d_3_loss/Mean_grad/DynamicStitchDynamicStitch:training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range8training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/mod:training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Fill*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*
N*
_output_shapes
:
Ќ
>training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: **
_class 
loc:@loss/conv2d_3_loss/Mean*
value	B :

<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/MaximumMaximumBtraining/Adam/gradients/loss/conv2d_3_loss/Mean_grad/DynamicStitch>training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum/y**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
:*
T0

=training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/floordivFloorDiv:training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum*
_output_shapes
:*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean
к
<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/ReshapeReshape>training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/truedivBtraining/Adam/gradients/loss/conv2d_3_loss/Mean_grad/DynamicStitch*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*
Tshape0
б
9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/TileTile<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Reshape=training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/floordiv*

Tmultiples0*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
О
<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_2Shapeloss/conv2d_3_loss/Abs**
_class 
loc:@loss/conv2d_3_loss/Mean*
out_type0*
_output_shapes
:*
T0
П
<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_3Shapeloss/conv2d_3_loss/Mean*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*
out_type0*
_output_shapes
:
А
:training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/ConstConst**
_class 
loc:@loss/conv2d_3_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
Ѕ
9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/ProdProd<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_2:training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
: 
В
<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:**
_class 
loc:@loss/conv2d_3_loss/Mean*
valueB: 
Љ
;training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Prod_1Prod<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_3<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Const_1**
_class 
loc:@loss/conv2d_3_loss/Mean*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Ў
@training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum_1/yConst**
_class 
loc:@loss/conv2d_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

>training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum_1Maximum;training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Prod_1@training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum_1/y*
_output_shapes
: *
T0**
_class 
loc:@loss/conv2d_3_loss/Mean

?training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/floordiv_1FloorDiv9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Prod>training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum_1*
_output_shapes
: *
T0**
_class 
loc:@loss/conv2d_3_loss/Mean
ю
9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/CastCast?training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/floordiv_1**
_class 
loc:@loss/conv2d_3_loss/Mean*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
Ѕ
<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/truedivRealDiv9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Tile9training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Cast*
T0**
_class 
loc:@loss/conv2d_3_loss/Mean*1
_output_shapes
:џџџџџџџџџИа
П
8training/Adam/gradients/loss/conv2d_3_loss/Abs_grad/SignSignloss/conv2d_3_loss/sub*
T0*)
_class
loc:@loss/conv2d_3_loss/Abs*1
_output_shapes
:џџџџџџџџџИа

7training/Adam/gradients/loss/conv2d_3_loss/Abs_grad/mulMul<training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/truediv8training/Adam/gradients/loss/conv2d_3_loss/Abs_grad/Sign*1
_output_shapes
:џџџџџџџџџИа*
T0*)
_class
loc:@loss/conv2d_3_loss/Abs
Д
9training/Adam/gradients/loss/conv2d_3_loss/sub_grad/ShapeShapeconv2d_3/BiasAdd*
T0*)
_class
loc:@loss/conv2d_3_loss/sub*
out_type0*
_output_shapes
:
Е
;training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape_1Shapeconv2d_3_target*
T0*)
_class
loc:@loss/conv2d_3_loss/sub*
out_type0*
_output_shapes
:
Т
Itraining/Adam/gradients/loss/conv2d_3_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape;training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape_1*
T0*)
_class
loc:@loss/conv2d_3_loss/sub*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
­
7training/Adam/gradients/loss/conv2d_3_loss/sub_grad/SumSum7training/Adam/gradients/loss/conv2d_3_loss/Abs_grad/mulItraining/Adam/gradients/loss/conv2d_3_loss/sub_grad/BroadcastGradientArgs*
T0*)
_class
loc:@loss/conv2d_3_loss/sub*
_output_shapes
:*
	keep_dims( *

Tidx0
Џ
;training/Adam/gradients/loss/conv2d_3_loss/sub_grad/ReshapeReshape7training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Sum9training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape*
T0*)
_class
loc:@loss/conv2d_3_loss/sub*
Tshape0*1
_output_shapes
:џџџџџџџџџИа
Б
9training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Sum_1Sum7training/Adam/gradients/loss/conv2d_3_loss/Abs_grad/mulKtraining/Adam/gradients/loss/conv2d_3_loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/conv2d_3_loss/sub
Ч
7training/Adam/gradients/loss/conv2d_3_loss/sub_grad/NegNeg9training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Sum_1*
T0*)
_class
loc:@loss/conv2d_3_loss/sub*
_output_shapes
:
Ь
=training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Reshape_1Reshape7training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Neg;training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape_1*)
_class
loc:@loss/conv2d_3_loss/sub*
Tshape0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0
ц
9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGrad;training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Reshape*
T0*#
_class
loc:@conv2d_3/BiasAdd*
data_formatNHWC*
_output_shapes
:
ь
8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNShapeN%up_sampling2d_1/ResizeNearestNeighborconv2d_3/kernel/read*'
_class
loc:@conv2d_3/convolution*
out_type0*
N* 
_output_shapes
::*
T0
М
Etraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNconv2d_3/kernel/read;training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Reshape*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа@*
	dilations
*
T0*'
_class
loc:@conv2d_3/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
Ц
Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_1/ResizeNearestNeighbor:training/Adam/gradients/conv2d_3/convolution_grad/ShapeN:1;training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Reshape*&
_output_shapes
:@*
	dilations
*
T0*'
_class
loc:@conv2d_3/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ь
atraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*
valueB"  (  
Ў
\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradEtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputatraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
T0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*1
_output_shapes
:џџџџџџџџџЈ@*
align_corners( 
Ю
;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradSwitch\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGraddropout_2/cond/pred_id*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@*
T0
о
training/Adam/gradients/SwitchSwitchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@
Ж
 training/Adam/gradients/IdentityIdentity training/Adam/gradients/Switch:1*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*1
_output_shapes
:џџџџџџџџџЈ@
Ћ
training/Adam/gradients/Shape_1Shape training/Adam/gradients/Switch:1*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
out_type0*
_output_shapes
:
З
#training/Adam/gradients/zeros/ConstConst!^training/Adam/gradients/Identity*
_output_shapes
: **
_class 
loc:@leaky_re_lu_2/LeakyRelu*
valueB
 *    *
dtype0
х
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*

index_type0*1
_output_shapes
:џџџџџџџџџЈ@

>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџЈ@: 
Ъ
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ShapeShapedropout_2/cond/dropout/truediv*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
out_type0*
_output_shapes
:
Ъ
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1Shapedropout_2/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
out_type0*
_output_shapes
:
в
Mtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1dropout_2/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџЈ@
Н
;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
П
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
Tshape0*1
_output_shapes
:џџџџџџџџџЈ@

=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Muldropout_2/cond/dropout/truediv=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1*1
_output_shapes
:џџџџџџџџџЈ@*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
У
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
Х
Atraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
Tshape0*1
_output_shapes
:џџџџџџџџџЈ@
Ц
Atraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/ShapeShapedropout_2/cond/mul*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
out_type0*
_output_shapes
:*
T0
Й
Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Shape_1Const*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
valueB *
dtype0*
_output_shapes
: 
т
Qtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsAtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/ShapeCtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDivRealDiv?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshapedropout_2/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџЈ@*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv
б
?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/SumSumCtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDivQtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
_output_shapes
:
Я
Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/ReshapeReshape?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/SumAtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Shape*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
Tshape0*1
_output_shapes
:џџџџџџџџџЈ@
Щ
?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/NegNegdropout_2/cond/mul*1
_output_shapes
:џџџџџџџџџЈ@*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv

Etraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_1RealDiv?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Negdropout_2/cond/dropout/sub*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@*
T0
Ђ
Etraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_2RealDivEtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_1dropout_2/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@
Н
?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/mulMul?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeEtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_2*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@
б
Atraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Sum_1Sum?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/mulStraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv
К
Etraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Reshape_1ReshapeAtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Sum_1Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
Tshape0*
_output_shapes
: 
З
5training/Adam/gradients/dropout_2/cond/mul_grad/ShapeShapedropout_2/cond/mul/Switch:1*
T0*%
_class
loc:@dropout_2/cond/mul*
out_type0*
_output_shapes
:
Ё
7training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1Const*
_output_shapes
: *%
_class
loc:@dropout_2/cond/mul*
valueB *
dtype0
В
Etraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_2/cond/mul_grad/Shape7training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_2/cond/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ј
3training/Adam/gradients/dropout_2/cond/mul_grad/MulMulCtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Reshapedropout_2/cond/mul/y*
T0*%
_class
loc:@dropout_2/cond/mul*1
_output_shapes
:џџџџџџџџџЈ@

3training/Adam/gradients/dropout_2/cond/mul_grad/SumSum3training/Adam/gradients/dropout_2/cond/mul_grad/MulEtraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs*
T0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0

7training/Adam/gradients/dropout_2/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_2/cond/mul_grad/Sum5training/Adam/gradients/dropout_2/cond/mul_grad/Shape*
T0*%
_class
loc:@dropout_2/cond/mul*
Tshape0*1
_output_shapes
:џџџџџџџџџЈ@

5training/Adam/gradients/dropout_2/cond/mul_grad/Mul_1Muldropout_2/cond/mul/Switch:1Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Reshape*
T0*%
_class
loc:@dropout_2/cond/mul*1
_output_shapes
:џџџџџџџџџЈ@
Ѓ
5training/Adam/gradients/dropout_2/cond/mul_grad/Sum_1Sum5training/Adam/gradients/dropout_2/cond/mul_grad/Mul_1Gtraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@dropout_2/cond/mul

9training/Adam/gradients/dropout_2/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_2/cond/mul_grad/Sum_17training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_2/cond/mul*
Tshape0*
_output_shapes
: 
р
 training/Adam/gradients/Switch_1Switchleaky_re_lu_2/LeakyReludropout_2/cond/pred_id*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu
И
"training/Adam/gradients/Identity_1Identity training/Adam/gradients/Switch_1*1
_output_shapes
:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu
Ћ
training/Adam/gradients/Shape_2Shape training/Adam/gradients/Switch_1**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
out_type0*
_output_shapes
:*
T0
Л
%training/Adam/gradients/zeros_1/ConstConst#^training/Adam/gradients/Identity_1*
dtype0*
_output_shapes
: **
_class 
loc:@leaky_re_lu_2/LeakyRelu*
valueB
 *    
щ
training/Adam/gradients/zeros_1Filltraining/Adam/gradients/Shape_2%training/Adam/gradients/zeros_1/Const*1
_output_shapes
:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*

index_type0

@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_17training/Adam/gradients/dropout_2/cond/mul_grad/Reshape*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџЈ@: 

training/Adam/gradients/AddNAddN>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_grad**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N*1
_output_shapes
:џџџџџџџџџЈ@*
T0
ћ
Btraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGradLeakyReluGradtraining/Adam/gradients/AddNconv2d_2/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
alpha%>*1
_output_shapes
:џџџџџџџџџЈ@
э
9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*#
_class
loc:@conv2d_2/BiasAdd*
data_formatNHWC*
_output_shapes
:@*
T0
о
8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNShapeNmax_pooling2d_1/MaxPoolconv2d_2/kernel/read*
T0*'
_class
loc:@conv2d_2/convolution*
out_type0*
N* 
_output_shapes
::
У
Etraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNconv2d_2/kernel/readBtraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ 
П
Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_1/MaxPool:training/Adam/gradients/conv2d_2/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*'
_class
loc:@conv2d_2/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @*
	dilations
*
T0
џ
@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGraddropout_1/cond/Mergemax_pooling2d_1/MaxPoolEtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput*
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
Є
;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradSwitch@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGraddropout_1/cond/pred_id*
T0**
_class 
loc:@max_pooling2d_1/MaxPool*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа 
р
 training/Adam/gradients/Switch_2Switchleaky_re_lu_1/LeakyReludropout_1/cond/pred_id*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа *
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu
К
"training/Adam/gradients/Identity_2Identity"training/Adam/gradients/Switch_2:1*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*1
_output_shapes
:џџџџџџџџџИа 
­
training/Adam/gradients/Shape_3Shape"training/Adam/gradients/Switch_2:1*
_output_shapes
:*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
out_type0
Л
%training/Adam/gradients/zeros_2/ConstConst#^training/Adam/gradients/Identity_2**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
valueB
 *    *
dtype0*
_output_shapes
: 
щ
training/Adam/gradients/zeros_2Filltraining/Adam/gradients/Shape_3%training/Adam/gradients/zeros_2/Const*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*

index_type0*1
_output_shapes
:џџџџџџџџџИа 

>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_2*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџИа : 
Ъ
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ShapeShapedropout_1/cond/dropout/truediv*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0*
_output_shapes
:
Ъ
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1Shapedropout_1/cond/dropout/Floor*
_output_shapes
:*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0
в
Mtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1dropout_1/cond/dropout/Floor*1
_output_shapes
:џџџџџџџџџИа *
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
Н
;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
П
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0*1
_output_shapes
:џџџџџџџџџИа 

=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Muldropout_1/cond/dropout/truediv=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1*1
_output_shapes
:џџџџџџџџџИа *
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
У
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:
Х
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*1
_output_shapes
:џџџџџџџџџИа *
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0
Ц
Atraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/ShapeShapedropout_1/cond/mul*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
out_type0*
_output_shapes
:
Й
Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1Const*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
valueB *
dtype0*
_output_shapes
: 
т
Qtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsAtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/ShapeCtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDivRealDiv?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshapedropout_1/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа 
б
?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/SumSumCtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDivQtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
Я
Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/ReshapeReshape?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/SumAtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Shape*1
_output_shapes
:џџџџџџџџџИа *
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
Tshape0
Щ
?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/NegNegdropout_1/cond/mul*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа 

Etraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1RealDiv?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Negdropout_1/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа 
Ђ
Etraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2RealDivEtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1dropout_1/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа 
Н
?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/mulMul?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeEtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа 
б
Atraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1Sum?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/mulStraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgs:1*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
К
Etraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Reshape_1ReshapeAtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
Tshape0*
_output_shapes
: 
З
5training/Adam/gradients/dropout_1/cond/mul_grad/ShapeShapedropout_1/cond/mul/Switch:1*%
_class
loc:@dropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
Ё
7training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1Const*%
_class
loc:@dropout_1/cond/mul*
valueB *
dtype0*
_output_shapes
: 
В
Etraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_1/cond/mul_grad/Shape7training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@dropout_1/cond/mul
ј
3training/Adam/gradients/dropout_1/cond/mul_grad/MulMulCtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Reshapedropout_1/cond/mul/y*1
_output_shapes
:џџџџџџџџџИа *
T0*%
_class
loc:@dropout_1/cond/mul

3training/Adam/gradients/dropout_1/cond/mul_grad/SumSum3training/Adam/gradients/dropout_1/cond/mul_grad/MulEtraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs*
T0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0

7training/Adam/gradients/dropout_1/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_1/cond/mul_grad/Sum5training/Adam/gradients/dropout_1/cond/mul_grad/Shape*
T0*%
_class
loc:@dropout_1/cond/mul*
Tshape0*1
_output_shapes
:џџџџџџџџџИа 

5training/Adam/gradients/dropout_1/cond/mul_grad/Mul_1Muldropout_1/cond/mul/Switch:1Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Reshape*1
_output_shapes
:џџџџџџџџџИа *
T0*%
_class
loc:@dropout_1/cond/mul
Ѓ
5training/Adam/gradients/dropout_1/cond/mul_grad/Sum_1Sum5training/Adam/gradients/dropout_1/cond/mul_grad/Mul_1Gtraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0

9training/Adam/gradients/dropout_1/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_1/cond/mul_grad/Sum_17training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1*%
_class
loc:@dropout_1/cond/mul*
Tshape0*
_output_shapes
: *
T0
р
 training/Adam/gradients/Switch_3Switchleaky_re_lu_1/LeakyReludropout_1/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*N
_output_shapes<
::џџџџџџџџџИа :џџџџџџџџџИа 
И
"training/Adam/gradients/Identity_3Identity training/Adam/gradients/Switch_3*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*1
_output_shapes
:џџџџџџџџџИа 
Ћ
training/Adam/gradients/Shape_4Shape training/Adam/gradients/Switch_3**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
out_type0*
_output_shapes
:*
T0
Л
%training/Adam/gradients/zeros_3/ConstConst#^training/Adam/gradients/Identity_3**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
valueB
 *    *
dtype0*
_output_shapes
: 
щ
training/Adam/gradients/zeros_3Filltraining/Adam/gradients/Shape_4%training/Adam/gradients/zeros_3/Const*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*

index_type0*1
_output_shapes
:џџџџџџџџџИа 

@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_37training/Adam/gradients/dropout_1/cond/mul_grad/Reshape**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџИа : *
T0

training/Adam/gradients/AddN_1AddN>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_grad*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
N*1
_output_shapes
:џџџџџџџџџИа 
§
Btraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGradLeakyReluGradtraining/Adam/gradients/AddN_1conv2d_1/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
alpha%>*1
_output_shapes
:џџџџџџџџџИа 
э
9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_1/BiasAdd*
data_formatNHWC*
_output_shapes
: 
е
8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNShapeNconv2d_1_inputconv2d_1/kernel/read*
T0*'
_class
loc:@conv2d_1/convolution*
out_type0*
N* 
_output_shapes
::
У
Etraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNconv2d_1/kernel/readBtraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа*
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
Ж
Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_1_input:training/Adam/gradients/conv2d_1/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*'
_class
loc:@conv2d_1/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0
_
training/Adam/AssignAdd/valueConst*
_output_shapes
: *
value	B	 R*
dtype0	
Ќ
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0	*"
_class
loc:@Adam/iterations
p
training/Adam/CastCastAdam/iterations/read*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0	
X
training/Adam/add/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
_output_shapes
: *
T0
^
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
_output_shapes
: *
T0
X
training/Adam/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_1Const*
valueB
 *  *
dtype0*
_output_shapes
: 
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
T0*
_output_shapes
: 

training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
_output_shapes
: *
T0
`
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
T0*
_output_shapes
: 
x
training/Adam/zerosConst*%
valueB *    *
dtype0*&
_output_shapes
: 

training/Adam/Variable
VariableV2*
dtype0*
	container *&
_output_shapes
: *
shape: *
shared_name 
й
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*&
_output_shapes
: 

training/Adam/Variable/readIdentitytraining/Adam/Variable*
T0*)
_class
loc:@training/Adam/Variable*&
_output_shapes
: 
b
training/Adam/zeros_1Const*
valueB *    *
dtype0*
_output_shapes
: 

training/Adam/Variable_1
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
shape: 
е
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
: 

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
: 
~
%training/Adam/zeros_2/shape_as_tensorConst*%
valueB"          @   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Є
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*
T0*

index_type0*&
_output_shapes
: @

training/Adam/Variable_2
VariableV2*
shared_name *
dtype0*
	container *&
_output_shapes
: @*
shape: @
с
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*&
_output_shapes
: @
Ё
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*&
_output_shapes
: @*
T0*+
_class!
loc:@training/Adam/Variable_2
b
training/Adam/zeros_3Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_3
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
е
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
T0*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
:@
~
%training/Adam/zeros_4/shape_as_tensorConst*%
valueB"      @      *
dtype0*
_output_shapes
:
`
training/Adam/zeros_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Є
training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*
T0*

index_type0*&
_output_shapes
:@

training/Adam/Variable_4
VariableV2*
shape:@*
shared_name *
dtype0*
	container *&
_output_shapes
:@
с
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*&
_output_shapes
:@
Ё
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4*&
_output_shapes
:@
b
training/Adam/zeros_5Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_5
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
е
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5

training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
:
z
training/Adam/zeros_6Const*&
_output_shapes
: *%
valueB *    *
dtype0

training/Adam/Variable_6
VariableV2*
dtype0*
	container *&
_output_shapes
: *
shape: *
shared_name 
с
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*&
_output_shapes
: 
Ё
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
T0*+
_class!
loc:@training/Adam/Variable_6*&
_output_shapes
: 
b
training/Adam/zeros_7Const*
valueB *    *
dtype0*
_output_shapes
: 

training/Adam/Variable_7
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
е
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes
: 

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
: 
~
%training/Adam/zeros_8/shape_as_tensorConst*%
valueB"          @   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_8/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Є
training/Adam/zeros_8Fill%training/Adam/zeros_8/shape_as_tensortraining/Adam/zeros_8/Const*&
_output_shapes
: @*
T0*

index_type0

training/Adam/Variable_8
VariableV2*
dtype0*
	container *&
_output_shapes
: @*
shape: @*
shared_name 
с
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*&
_output_shapes
: @*
use_locking(
Ё
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*&
_output_shapes
: @*
T0*+
_class!
loc:@training/Adam/Variable_8
b
training/Adam/zeros_9Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_9
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
е
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(

training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
:@

&training/Adam/zeros_10/shape_as_tensorConst*
_output_shapes
:*%
valueB"      @      *
dtype0
a
training/Adam/zeros_10/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ї
training/Adam/zeros_10Fill&training/Adam/zeros_10/shape_as_tensortraining/Adam/zeros_10/Const*

index_type0*&
_output_shapes
:@*
T0

training/Adam/Variable_10
VariableV2*
shared_name *
dtype0*
	container *&
_output_shapes
:@*
shape:@
х
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10
Є
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*&
_output_shapes
:@*
T0
c
training/Adam/zeros_11Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_11
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
й
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:

training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
:
p
&training/Adam/zeros_12/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_12/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_12Fill&training/Adam/zeros_12/shape_as_tensortraining/Adam/zeros_12/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_12
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
й
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes
:

training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes
:
p
&training/Adam/zeros_13/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_13/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_13Fill&training/Adam/zeros_13/shape_as_tensortraining/Adam/zeros_13/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_13
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
й
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:

training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
T0*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
:
p
&training/Adam/zeros_14/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_14/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_14
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes
:

training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes
:
p
&training/Adam/zeros_15/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_15/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_15Fill&training/Adam/zeros_15/shape_as_tensortraining/Adam/zeros_15/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_15
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
й
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
:
p
&training/Adam/zeros_16/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_16
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
й
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*
_output_shapes
:

training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_16
p
&training/Adam/zeros_17/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_17/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_17Fill&training/Adam/zeros_17/shape_as_tensortraining/Adam/zeros_17/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_17
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
й
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:

training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
_output_shapes
:
z
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
T0*&
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ј
training/Adam/mul_2Multraining/Adam/sub_2Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: *
T0
u
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*&
_output_shapes
: *
T0
|
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_6/read*
T0*&
_output_shapes
: 
Z
training/Adam/sub_3/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/SquareSquareFtraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
v
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*&
_output_shapes
: 
u
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*&
_output_shapes
: 
s
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*&
_output_shapes
: 
Z
training/Adam/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_3Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*
T0*&
_output_shapes
: 

training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
T0*&
_output_shapes
: 
l
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*&
_output_shapes
: 
Z
training/Adam/add_3/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
x
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*&
_output_shapes
: 
}
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0*&
_output_shapes
: 
z
training/Adam/sub_4Subconv2d_1/kernel/readtraining/Adam/truediv_1*
T0*&
_output_shapes
: 
а
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*)
_class
loc:@training/Adam/Variable*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0
ж
training/Adam/Assign_1Assigntraining/Adam/Variable_6training/Adam/add_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*&
_output_shapes
: 
Ф
training/Adam/Assign_2Assignconv2d_1/kerneltraining/Adam/sub_4*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
_output_shapes
: *
T0
Z
training/Adam/sub_5/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_7Multraining/Adam/sub_59training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
_output_shapes
: *
T0
p
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_7/read*
T0*
_output_shapes
: 
Z
training/Adam/sub_6/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_1Square9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes
: 
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
_output_shapes
: *
T0
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
: 
Z
training/Adam/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_5Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
T0*
_output_shapes
: 

training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
T0*
_output_shapes
: 
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
: 
Z
training/Adam/add_6/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
_output_shapes
: *
T0
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes
: 
l
training/Adam/sub_7Subconv2d_1/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes
: 
Ъ
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
: 
Ъ
training/Adam/Assign_4Assigntraining/Adam/Variable_7training/Adam/add_5*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7
Д
training/Adam/Assign_5Assignconv2d_1/biastraining/Adam/sub_7*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
: 
}
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*&
_output_shapes
: @*
T0
Z
training/Adam/sub_8/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
T0*
_output_shapes
: 
Љ
training/Adam/mul_12Multraining/Adam/sub_8Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
w
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*&
_output_shapes
: @
}
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_8/read*&
_output_shapes
: @*
T0
Z
training/Adam/sub_9/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_2SquareFtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
y
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*&
_output_shapes
: @
w
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*&
_output_shapes
: @*
T0
t
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*&
_output_shapes
: @*
T0
Z
training/Adam/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_7Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*&
_output_shapes
: @*
T0

training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
T0*&
_output_shapes
: @
l
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*&
_output_shapes
: @
Z
training/Adam/add_9/yConst*
dtype0*
_output_shapes
: *
valueB
 *Пж3
x
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0*&
_output_shapes
: @
~
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*&
_output_shapes
: @
{
training/Adam/sub_10Subconv2d_2/kernel/readtraining/Adam/truediv_3*&
_output_shapes
: @*
T0
ж
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*&
_output_shapes
: @*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(
ж
training/Adam/Assign_7Assigntraining/Adam/Variable_8training/Adam/add_8*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0
Х
training/Adam/Assign_8Assignconv2d_2/kerneltraining/Adam/sub_10*&
_output_shapes
: @*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(
q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
_output_shapes
:@*
T0
[
training/Adam/sub_11/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_17Multraining/Adam/sub_119training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
:@
q
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_9/read*
T0*
_output_shapes
:@
[
training/Adam/sub_12/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_3Square9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes
:@
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
:@
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
_output_shapes
:@*
T0
Z
training/Adam/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_9Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
T0*
_output_shapes
:@

training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
T0*
_output_shapes
:@
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes
:@
[
training/Adam/add_12/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
_output_shapes
:@*
T0
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes
:@
m
training/Adam/sub_13Subconv2d_2/bias/readtraining/Adam/truediv_4*
T0*
_output_shapes
:@
Ы
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
Ь
training/Adam/Assign_10Assigntraining/Adam/Variable_9training/Adam/add_11*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
Ж
training/Adam/Assign_11Assignconv2d_2/biastraining/Adam/sub_13*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@
}
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*&
_output_shapes
:@*
T0
[
training/Adam/sub_14/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
_output_shapes
: *
T0
Њ
training/Adam/mul_22Multraining/Adam/sub_14Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
x
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*&
_output_shapes
:@*
T0
~
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0*&
_output_shapes
:@
[
training/Adam/sub_15/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_15Subtraining/Adam/sub_15/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_4SquareFtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
z
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*&
_output_shapes
:@
x
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*&
_output_shapes
:@*
T0
u
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*&
_output_shapes
:@*
T0
[
training/Adam/Const_10Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_11Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*
T0*&
_output_shapes
:@

training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*&
_output_shapes
:@*
T0
l
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*&
_output_shapes
:@
[
training/Adam/add_15/yConst*
dtype0*
_output_shapes
: *
valueB
 *Пж3
z
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*&
_output_shapes
:@

training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*&
_output_shapes
:@*
T0
{
training/Adam/sub_16Subconv2d_3/kernel/readtraining/Adam/truediv_5*
T0*&
_output_shapes
:@
и
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4
к
training/Adam/Assign_13Assigntraining/Adam/Variable_10training/Adam/add_14*&
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(
Ц
training/Adam/Assign_14Assignconv2d_3/kerneltraining/Adam/sub_16*"
_class
loc:@conv2d_3/kernel*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
q
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
_output_shapes
:*
T0
[
training/Adam/sub_17/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_27Multraining/Adam/sub_179training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
_output_shapes
:*
T0
r
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_11/read*
T0*
_output_shapes
:
[
training/Adam/sub_18/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_5Square9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
_output_shapes
:*
T0
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes
:
i
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes
:
[
training/Adam/Const_12Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_13Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
T0*
_output_shapes
:

training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
_output_shapes
:*
T0
`
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes
:
[
training/Adam/add_18/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
n
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
_output_shapes
:*
T0
s
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
_output_shapes
:*
T0
m
training/Adam/sub_19Subconv2d_3/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes
:
Ь
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(
Ю
training/Adam/Assign_16Assigntraining/Adam/Variable_11training/Adam/add_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:
Ж
training/Adam/Assign_17Assignconv2d_3/biastraining/Adam/sub_19*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes
:

training/group_depsNoOp	^loss/mul^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_2^training/Adam/Assign_3^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9


group_depsNoOp	^loss/mul

IsVariableInitializedIsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_2IsVariableInitializedconv2d_2/kernel*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel*
dtype0

IsVariableInitialized_3IsVariableInitializedconv2d_2/bias*
_output_shapes
: * 
_class
loc:@conv2d_2/bias*
dtype0

IsVariableInitialized_4IsVariableInitializedconv2d_3/kernel*
_output_shapes
: *"
_class
loc:@conv2d_3/kernel*
dtype0

IsVariableInitialized_5IsVariableInitializedconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_6IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
z
IsVariableInitialized_7IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializedAdam/beta_1*
_output_shapes
: *
_class
loc:@Adam/beta_1*
dtype0

IsVariableInitialized_9IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 

IsVariableInitialized_11IsVariableInitializedtraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
dtype0*
_output_shapes
: 

IsVariableInitialized_12IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_13IsVariableInitializedtraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_14IsVariableInitializedtraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0*
_output_shapes
: 

IsVariableInitialized_15IsVariableInitializedtraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0*
_output_shapes
: 

IsVariableInitialized_16IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
: 

IsVariableInitialized_17IsVariableInitializedtraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes
: 

IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0*
_output_shapes
: 

IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes
: 

IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_10*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_10*
dtype0

IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
: 

IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
: 

IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
dtype0*
_output_shapes
: 

IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_14*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_14*
dtype0

IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
dtype0*
_output_shapes
: 

IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0*
_output_shapes
: 

IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_17*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_17*
dtype0
ю
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^conv2d_3/bias/Assign^conv2d_3/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign ^training/Adam/Variable_2/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign""Ќ
trainable_variables
`
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02conv2d_1/random_uniform:08
Q
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02conv2d_1/Const:08
`
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02conv2d_2/random_uniform:08
Q
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:02conv2d_2/Const:08
`
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:02conv2d_3/random_uniform:08
Q
conv2d_3/bias:0conv2d_3/bias/Assignconv2d_3/bias/read:02conv2d_3/Const:08
f
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:08
F
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:08
V
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:08
V
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:08
R
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:08
q
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:02training/Adam/zeros:08
y
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:02training/Adam/zeros_1:08
y
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:02training/Adam/zeros_2:08
y
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:02training/Adam/zeros_3:08
y
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:02training/Adam/zeros_4:08
y
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:02training/Adam/zeros_5:08
y
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:02training/Adam/zeros_6:08
y
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:02training/Adam/zeros_7:08
y
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:02training/Adam/zeros_8:08
y
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:02training/Adam/zeros_9:08
}
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:02training/Adam/zeros_10:08
}
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:02training/Adam/zeros_11:08
}
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:02training/Adam/zeros_12:08
}
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:02training/Adam/zeros_13:08
}
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:02training/Adam/zeros_14:08
}
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/zeros_15:08
}
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign training/Adam/Variable_16/read:02training/Adam/zeros_16:08
}
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign training/Adam/Variable_17/read:02training/Adam/zeros_17:08"
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
leaky_re_lu_1/LeakyRelu:08
leaky_re_lu_1/LeakyRelu:0dropout_1/cond/mul/Switch:14
dropout_1/cond/pred_id:0dropout_1/cond/pred_id:0
Ш
dropout_1/cond/cond_text_1dropout_1/cond/pred_id:0dropout_1/cond/switch_f:0*є
dropout_1/cond/Switch_1:0
dropout_1/cond/Switch_1:1
dropout_1/cond/pred_id:0
dropout_1/cond/switch_f:0
leaky_re_lu_1/LeakyRelu:06
leaky_re_lu_1/LeakyRelu:0dropout_1/cond/Switch_1:04
dropout_1/cond/pred_id:0dropout_1/cond/pred_id:0
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
leaky_re_lu_2/LeakyRelu:08
leaky_re_lu_2/LeakyRelu:0dropout_2/cond/mul/Switch:14
dropout_2/cond/pred_id:0dropout_2/cond/pred_id:0
Ш
dropout_2/cond/cond_text_1dropout_2/cond/pred_id:0dropout_2/cond/switch_f:0*є
dropout_2/cond/Switch_1:0
dropout_2/cond/Switch_1:1
dropout_2/cond/pred_id:0
dropout_2/cond/switch_f:0
leaky_re_lu_2/LeakyRelu:06
leaky_re_lu_2/LeakyRelu:0dropout_2/cond/Switch_1:04
dropout_2/cond/pred_id:0dropout_2/cond/pred_id:0"Ђ
	variables
`
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02conv2d_1/random_uniform:08
Q
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02conv2d_1/Const:08
`
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02conv2d_2/random_uniform:08
Q
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:02conv2d_2/Const:08
`
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:02conv2d_3/random_uniform:08
Q
conv2d_3/bias:0conv2d_3/bias/Assignconv2d_3/bias/read:02conv2d_3/Const:08
f
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:08
F
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:08
V
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:08
V
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:08
R
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:08
q
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:02training/Adam/zeros:08
y
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:02training/Adam/zeros_1:08
y
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:02training/Adam/zeros_2:08
y
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:02training/Adam/zeros_3:08
y
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:02training/Adam/zeros_4:08
y
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:02training/Adam/zeros_5:08
y
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:02training/Adam/zeros_6:08
y
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:02training/Adam/zeros_7:08
y
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:02training/Adam/zeros_8:08
y
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:02training/Adam/zeros_9:08
}
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:02training/Adam/zeros_10:08
}
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:02training/Adam/zeros_11:08
}
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:02training/Adam/zeros_12:08
}
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:02training/Adam/zeros_13:08
}
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:02training/Adam/zeros_14:08
}
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/zeros_15:08
}
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign training/Adam/Variable_16/read:02training/Adam/zeros_16:08
}
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign training/Adam/Variable_17/read:02training/Adam/zeros_17:08L