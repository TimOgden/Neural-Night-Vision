       ЃK"	  @ыЙ зAbrain.Event:2ЦЯ^џ     Йс	u[AыЙ зA"ђЃ

conv2d_1_inputPlaceholder*
dtype0*1
_output_shapes
:џџџџџџџџџИа*&
shape:џџџџџџџџџИа
v
conv2d_1/random_uniform/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
`
conv2d_1/random_uniform/minConst*
valueB
 *8JЬН*
dtype0*
_output_shapes
: 
`
conv2d_1/random_uniform/maxConst*
valueB
 *8JЬ=*
dtype0*
_output_shapes
: 
В
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
seedБџх)*
T0*
dtype0*&
_output_shapes
:@*
seed2ШЄЧ
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
T0*
_output_shapes
: 

conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*&
_output_shapes
:@*
T0

conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*
T0*&
_output_shapes
:@

conv2d_1/kernel
VariableV2*
dtype0*&
_output_shapes
:@*
	container *
shape:@*
shared_name 
Ш
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@

conv2d_1/kernel/readIdentityconv2d_1/kernel*&
_output_shapes
:@*
T0*"
_class
loc:@conv2d_1/kernel
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
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:@
t
conv2d_1/bias/readIdentityconv2d_1/bias*
_output_shapes
:@*
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
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа@*
	dilations
*
T0

conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа@

leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_1/BiasAdd*
T0*
alpha%>*1
_output_shapes
:џџџџџџџџџИа@
v
conv2d_2/random_uniform/shapeConst*
_output_shapes
:*%
valueB"      @   @   *
dtype0
`
conv2d_2/random_uniform/minConst*
_output_shapes
: *
valueB
 *:ЭН*
dtype0
`
conv2d_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *:Э=
В
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
dtype0*&
_output_shapes
:@@*
seed2ИЕ*
seedБџх)*
T0
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
_output_shapes
: *
T0

conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*&
_output_shapes
:@@*
T0

conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*&
_output_shapes
:@@*
T0

conv2d_2/kernel
VariableV2*
dtype0*&
_output_shapes
:@@*
	container *
shape:@@*
shared_name 
Ш
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:@@

conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
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
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
t
conv2d_2/bias/readIdentityconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0
s
"conv2d_2/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
ї
conv2d_2/convolutionConv2Dleaky_re_lu_1/LeakyReluconv2d_2/kernel/read*1
_output_shapes
:џџџџџџџџџИа@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа@

leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_2/BiasAdd*
alpha%>*1
_output_shapes
:џџџџџџџџџИа@*
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
dropout_1/keras_learning_phasePlaceholderWithDefault$dropout_1/keras_learning_phase/input*
_output_shapes
: *
shape: *
dtype0

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
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
T0
*
_output_shapes
: 
c
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
: *
T0

s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?

dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*
T0*1
_output_shapes
:џџџџџџџџџИа@
й
dropout_1/cond/mul/SwitchSwitchleaky_re_lu_2/LeakyReludropout_1/cond/pred_id*N
_output_shapes<
::џџџџџџџџџИа@:џџџџџџџџџИа@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu
z
dropout_1/cond/dropout/rateConst^dropout_1/cond/switch_t*
valueB
 *ЭЬЬ>*
dtype0*
_output_shapes
: 
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
)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ъ
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
dtype0*1
_output_shapes
:џџџџџџџџџИа@*
seed2иЂ*
seedБџх)*
T0
Ї
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Ь
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*1
_output_shapes
:џџџџџџџџџИа@
О
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*1
_output_shapes
:џџџџџџџџџИа@
 
dropout_1/cond/dropout/addAdddropout_1/cond/dropout/sub%dropout_1/cond/dropout/random_uniform*
T0*1
_output_shapes
:џџџџџџџџџИа@
}
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*1
_output_shapes
:џџџџџџџџџИа@*
T0

dropout_1/cond/dropout/truedivRealDivdropout_1/cond/muldropout_1/cond/dropout/sub*
T0*1
_output_shapes
:џџџџџџџџџИа@

dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/truedivdropout_1/cond/dropout/Floor*
T0*1
_output_shapes
:џџџџџџџџџИа@
з
dropout_1/cond/Switch_1Switchleaky_re_lu_2/LeakyReludropout_1/cond/pred_id*N
_output_shapes<
::џџџџџџџџџИа@:џџџџџџџџџИа@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu

dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
T0*
N*3
_output_shapes!
:џџџџџџџџџИа@: 
Ц
max_pooling2d_1/MaxPoolMaxPooldropout_1/cond/Merge*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ@*
T0*
data_formatNHWC*
strides

v
conv2d_3/random_uniform/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
`
conv2d_3/random_uniform/minConst*
valueB
 *я[qН*
dtype0*
_output_shapes
: 
`
conv2d_3/random_uniform/maxConst*
valueB
 *я[q=*
dtype0*
_output_shapes
: 
Г
%conv2d_3/random_uniform/RandomUniformRandomUniformconv2d_3/random_uniform/shape*
dtype0*'
_output_shapes
:@*
seed2єј*
seedБџх)*
T0
}
conv2d_3/random_uniform/subSubconv2d_3/random_uniform/maxconv2d_3/random_uniform/min*
_output_shapes
: *
T0

conv2d_3/random_uniform/mulMul%conv2d_3/random_uniform/RandomUniformconv2d_3/random_uniform/sub*'
_output_shapes
:@*
T0

conv2d_3/random_uniformAddconv2d_3/random_uniform/mulconv2d_3/random_uniform/min*
T0*'
_output_shapes
:@

conv2d_3/kernel
VariableV2*
shared_name *
dtype0*'
_output_shapes
:@*
	container *
shape:@
Щ
conv2d_3/kernel/AssignAssignconv2d_3/kernelconv2d_3/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(*'
_output_shapes
:@

conv2d_3/kernel/readIdentityconv2d_3/kernel*
T0*"
_class
loc:@conv2d_3/kernel*'
_output_shapes
:@
]
conv2d_3/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_3/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
shared_name *
dtype0
Ў
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/Const*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes	
:
u
conv2d_3/bias/readIdentityconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
_output_shapes	
:*
T0
s
"conv2d_3/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ј
conv2d_3/convolutionConv2Dmax_pooling2d_1/MaxPoolconv2d_3/kernel/read*2
_output_shapes 
:џџџџџџџџџЈ*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME

conv2d_3/BiasAddBiasAddconv2d_3/convolutionconv2d_3/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:џџџџџџџџџЈ

leaky_re_lu_3/LeakyRelu	LeakyReluconv2d_3/BiasAdd*
T0*
alpha%>*2
_output_shapes 
:џџџџџџџџџЈ
v
conv2d_4/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
`
conv2d_4/random_uniform/minConst*
_output_shapes
: *
valueB
 *ьQН*
dtype0
`
conv2d_4/random_uniform/maxConst*
valueB
 *ьQ=*
dtype0*
_output_shapes
: 
Г
%conv2d_4/random_uniform/RandomUniformRandomUniformconv2d_4/random_uniform/shape*
dtype0*(
_output_shapes
:*
seed2ЩЧ+*
seedБџх)*
T0
}
conv2d_4/random_uniform/subSubconv2d_4/random_uniform/maxconv2d_4/random_uniform/min*
_output_shapes
: *
T0

conv2d_4/random_uniform/mulMul%conv2d_4/random_uniform/RandomUniformconv2d_4/random_uniform/sub*(
_output_shapes
:*
T0

conv2d_4/random_uniformAddconv2d_4/random_uniform/mulconv2d_4/random_uniform/min*
T0*(
_output_shapes
:

conv2d_4/kernel
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
Ъ
conv2d_4/kernel/AssignAssignconv2d_4/kernelconv2d_4/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*(
_output_shapes
:

conv2d_4/kernel/readIdentityconv2d_4/kernel*(
_output_shapes
:*
T0*"
_class
loc:@conv2d_4/kernel
]
conv2d_4/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
{
conv2d_4/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ў
conv2d_4/bias/AssignAssignconv2d_4/biasconv2d_4/Const*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(
u
conv2d_4/bias/readIdentityconv2d_4/bias*
T0* 
_class
loc:@conv2d_4/bias*
_output_shapes	
:
s
"conv2d_4/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ј
conv2d_4/convolutionConv2Dleaky_re_lu_3/LeakyReluconv2d_4/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:џџџџџџџџџЈ

conv2d_4/BiasAddBiasAddconv2d_4/convolutionconv2d_4/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:џџџџџџџџџЈ

leaky_re_lu_4/LeakyRelu	LeakyReluconv2d_4/BiasAdd*
T0*
alpha%>*2
_output_shapes 
:џџџџџџџџџЈ

dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes
: : *
T0

]
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
T0
*
_output_shapes
: 
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

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*
T0*2
_output_shapes 
:џџџџџџџџџЈ
л
dropout_2/cond/mul/SwitchSwitchleaky_re_lu_4/LeakyReludropout_2/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*P
_output_shapes>
<:џџџџџџџџџЈ:џџџџџџџџџЈ
z
dropout_2/cond/dropout/rateConst^dropout_2/cond/switch_t*
valueB
 *ЭЬЬ>*
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
)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    

)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
_output_shapes
: *
valueB
 *  ?*
dtype0
Ъ
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
T0*
dtype0*2
_output_shapes 
:џџџџџџџџџЈ*
seed2Бњ*
seedБџх)
Ї
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Э
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*
T0*2
_output_shapes 
:џџџџџџџџџЈ
П
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*2
_output_shapes 
:џџџџџџџџџЈ*
T0
Ё
dropout_2/cond/dropout/addAdddropout_2/cond/dropout/sub%dropout_2/cond/dropout/random_uniform*
T0*2
_output_shapes 
:џџџџџџџџџЈ
~
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*
T0*2
_output_shapes 
:џџџџџџџџџЈ

dropout_2/cond/dropout/truedivRealDivdropout_2/cond/muldropout_2/cond/dropout/sub*2
_output_shapes 
:џџџџџџџџџЈ*
T0

dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/truedivdropout_2/cond/dropout/Floor*
T0*2
_output_shapes 
:џџџџџџџџџЈ
й
dropout_2/cond/Switch_1Switchleaky_re_lu_4/LeakyReludropout_2/cond/pred_id*P
_output_shapes>
<:џџџџџџџџџЈ:џџџџџџџџџЈ*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu

dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*4
_output_shapes"
 :џџџџџџџџџЈ: *
T0*
N
Ч
max_pooling2d_2/MaxPoolMaxPooldropout_2/cond/Merge*
ksize
*
paddingSAME*2
_output_shapes 
:џџџџџџџџџ*
T0*
data_formatNHWC*
strides

l
up_sampling2d_1/ShapeShapemax_pooling2d_2/MaxPool*
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
up_sampling2d_1/ConstConst*
_output_shapes
:*
valueB"      *
dtype0
u
up_sampling2d_1/mulMulup_sampling2d_1/strided_sliceup_sampling2d_1/Const*
T0*
_output_shapes
:
О
%up_sampling2d_1/ResizeNearestNeighborResizeNearestNeighbormax_pooling2d_2/MaxPoolup_sampling2d_1/mul*2
_output_shapes 
:џџџџџџџџџЈ*
align_corners( *
T0
v
conv2d_5/random_uniform/shapeConst*
_output_shapes
:*%
valueB"         @   *
dtype0
`
conv2d_5/random_uniform/minConst*
_output_shapes
: *
valueB
 *я[qН*
dtype0
`
conv2d_5/random_uniform/maxConst*
valueB
 *я[q=*
dtype0*
_output_shapes
: 
Г
%conv2d_5/random_uniform/RandomUniformRandomUniformconv2d_5/random_uniform/shape*'
_output_shapes
:@*
seed2Џ*
seedБџх)*
T0*
dtype0
}
conv2d_5/random_uniform/subSubconv2d_5/random_uniform/maxconv2d_5/random_uniform/min*
T0*
_output_shapes
: 

conv2d_5/random_uniform/mulMul%conv2d_5/random_uniform/RandomUniformconv2d_5/random_uniform/sub*
T0*'
_output_shapes
:@

conv2d_5/random_uniformAddconv2d_5/random_uniform/mulconv2d_5/random_uniform/min*
T0*'
_output_shapes
:@

conv2d_5/kernel
VariableV2*
dtype0*'
_output_shapes
:@*
	container *
shape:@*
shared_name 
Щ
conv2d_5/kernel/AssignAssignconv2d_5/kernelconv2d_5/random_uniform*"
_class
loc:@conv2d_5/kernel*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0

conv2d_5/kernel/readIdentityconv2d_5/kernel*
T0*"
_class
loc:@conv2d_5/kernel*'
_output_shapes
:@
[
conv2d_5/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_5/bias
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
­
conv2d_5/bias/AssignAssignconv2d_5/biasconv2d_5/Const*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias
t
conv2d_5/bias/readIdentityconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
_output_shapes
:@*
T0
s
"conv2d_5/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

conv2d_5/convolutionConv2D%up_sampling2d_1/ResizeNearestNeighborconv2d_5/kernel/read*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

conv2d_5/BiasAddBiasAddconv2d_5/convolutionconv2d_5/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџЈ@

leaky_re_lu_5/LeakyRelu	LeakyReluconv2d_5/BiasAdd*
T0*
alpha%>*1
_output_shapes
:џџџџџџџџџЈ@
v
conv2d_6/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
`
conv2d_6/random_uniform/minConst*
_output_shapes
: *
valueB
 *:ЭН*
dtype0
`
conv2d_6/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *:Э=
В
%conv2d_6/random_uniform/RandomUniformRandomUniformconv2d_6/random_uniform/shape*
dtype0*&
_output_shapes
:@@*
seed2ЧІ*
seedБџх)*
T0
}
conv2d_6/random_uniform/subSubconv2d_6/random_uniform/maxconv2d_6/random_uniform/min*
T0*
_output_shapes
: 

conv2d_6/random_uniform/mulMul%conv2d_6/random_uniform/RandomUniformconv2d_6/random_uniform/sub*
T0*&
_output_shapes
:@@

conv2d_6/random_uniformAddconv2d_6/random_uniform/mulconv2d_6/random_uniform/min*&
_output_shapes
:@@*
T0

conv2d_6/kernel
VariableV2*&
_output_shapes
:@@*
	container *
shape:@@*
shared_name *
dtype0
Ш
conv2d_6/kernel/AssignAssignconv2d_6/kernelconv2d_6/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*&
_output_shapes
:@@

conv2d_6/kernel/readIdentityconv2d_6/kernel*
T0*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:@@
[
conv2d_6/ConstConst*
_output_shapes
:@*
valueB@*    *
dtype0
y
conv2d_6/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
­
conv2d_6/bias/AssignAssignconv2d_6/biasconv2d_6/Const*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@conv2d_6/bias
t
conv2d_6/bias/readIdentityconv2d_6/bias*
T0* 
_class
loc:@conv2d_6/bias*
_output_shapes
:@
s
"conv2d_6/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ї
conv2d_6/convolutionConv2Dleaky_re_lu_5/LeakyReluconv2d_6/kernel/read*
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
conv2d_6/BiasAddBiasAddconv2d_6/convolutionconv2d_6/bias/read*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџЈ@*
T0

leaky_re_lu_6/LeakyRelu	LeakyReluconv2d_6/BiasAdd*
alpha%>*1
_output_shapes
:џџџџџџџџџЈ@*
T0

dropout_3/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes
: : *
T0

]
dropout_3/cond/switch_tIdentitydropout_3/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_3/cond/switch_fIdentitydropout_3/cond/Switch*
_output_shapes
: *
T0

c
dropout_3/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
: 
s
dropout_3/cond/mul/yConst^dropout_3/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 

dropout_3/cond/mulMuldropout_3/cond/mul/Switch:1dropout_3/cond/mul/y*
T0*1
_output_shapes
:џџџџџџџџџЈ@
й
dropout_3/cond/mul/SwitchSwitchleaky_re_lu_6/LeakyReludropout_3/cond/pred_id*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu
z
dropout_3/cond/dropout/rateConst^dropout_3/cond/switch_t*
valueB
 *ЭЬЬ>*
dtype0*
_output_shapes
: 
n
dropout_3/cond/dropout/ShapeShapedropout_3/cond/mul*
T0*
out_type0*
_output_shapes
:
{
dropout_3/cond/dropout/sub/xConst^dropout_3/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
}
dropout_3/cond/dropout/subSubdropout_3/cond/dropout/sub/xdropout_3/cond/dropout/rate*
T0*
_output_shapes
: 

)dropout_3/cond/dropout/random_uniform/minConst^dropout_3/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_3/cond/dropout/random_uniform/maxConst^dropout_3/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ъ
3dropout_3/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_3/cond/dropout/Shape*
seedБџх)*
T0*
dtype0*1
_output_shapes
:џџџџџџџџџЈ@*
seed2хо
Ї
)dropout_3/cond/dropout/random_uniform/subSub)dropout_3/cond/dropout/random_uniform/max)dropout_3/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ь
)dropout_3/cond/dropout/random_uniform/mulMul3dropout_3/cond/dropout/random_uniform/RandomUniform)dropout_3/cond/dropout/random_uniform/sub*1
_output_shapes
:џџџџџџџџџЈ@*
T0
О
%dropout_3/cond/dropout/random_uniformAdd)dropout_3/cond/dropout/random_uniform/mul)dropout_3/cond/dropout/random_uniform/min*1
_output_shapes
:џџџџџџџџџЈ@*
T0
 
dropout_3/cond/dropout/addAdddropout_3/cond/dropout/sub%dropout_3/cond/dropout/random_uniform*1
_output_shapes
:џџџџџџџџџЈ@*
T0
}
dropout_3/cond/dropout/FloorFloordropout_3/cond/dropout/add*1
_output_shapes
:џџџџџџџџџЈ@*
T0

dropout_3/cond/dropout/truedivRealDivdropout_3/cond/muldropout_3/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџЈ@*
T0

dropout_3/cond/dropout/mulMuldropout_3/cond/dropout/truedivdropout_3/cond/dropout/Floor*
T0*1
_output_shapes
:џџџџџџџџџЈ@
з
dropout_3/cond/Switch_1Switchleaky_re_lu_6/LeakyReludropout_3/cond/pred_id**
_class 
loc:@leaky_re_lu_6/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@*
T0

dropout_3/cond/MergeMergedropout_3/cond/Switch_1dropout_3/cond/dropout/mul*3
_output_shapes!
:џџџџџџџџџЈ@: *
T0*
N
v
conv2d_7/random_uniform/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
`
conv2d_7/random_uniform/minConst*
valueB
 *мО*
dtype0*
_output_shapes
: 
`
conv2d_7/random_uniform/maxConst*
valueB
 *м>*
dtype0*
_output_shapes
: 
В
%conv2d_7/random_uniform/RandomUniformRandomUniformconv2d_7/random_uniform/shape*&
_output_shapes
:@*
seed2юи*
seedБџх)*
T0*
dtype0
}
conv2d_7/random_uniform/subSubconv2d_7/random_uniform/maxconv2d_7/random_uniform/min*
_output_shapes
: *
T0

conv2d_7/random_uniform/mulMul%conv2d_7/random_uniform/RandomUniformconv2d_7/random_uniform/sub*
T0*&
_output_shapes
:@

conv2d_7/random_uniformAddconv2d_7/random_uniform/mulconv2d_7/random_uniform/min*&
_output_shapes
:@*
T0

conv2d_7/kernel
VariableV2*
shape:@*
shared_name *
dtype0*&
_output_shapes
:@*
	container 
Ш
conv2d_7/kernel/AssignAssignconv2d_7/kernelconv2d_7/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_7/kernel*
validate_shape(*&
_output_shapes
:@

conv2d_7/kernel/readIdentityconv2d_7/kernel*
T0*"
_class
loc:@conv2d_7/kernel*&
_output_shapes
:@
[
conv2d_7/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv2d_7/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
­
conv2d_7/bias/AssignAssignconv2d_7/biasconv2d_7/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv2d_7/bias
t
conv2d_7/bias/readIdentityconv2d_7/bias*
T0* 
_class
loc:@conv2d_7/bias*
_output_shapes
:
s
"conv2d_7/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
є
conv2d_7/convolutionConv2Ddropout_3/cond/Mergeconv2d_7/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ

conv2d_7/BiasAddBiasAddconv2d_7/convolutionconv2d_7/bias/read*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџЈ*
T0

lambda_1/DepthToSpaceDepthToSpaceconv2d_7/BiasAdd*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа*

block_size
m
lambda_1/ResizeBilinear/sizeConst*
valueB"8  P  *
dtype0*
_output_shapes
:
Џ
lambda_1/ResizeBilinearResizeBilinearlambda_1/DepthToSpacelambda_1/ResizeBilinear/size*1
_output_shapes
:џџџџџџџџџИа*
align_corners( *
T0

lambda_1/PlaceholderPlaceholder*
dtype0*1
_output_shapes
:џџџџџџџџџЈ*&
shape:џџџџџџџџџЈ
Ђ
lambda_1/DepthToSpace_1DepthToSpacelambda_1/Placeholder*

block_size*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа
o
lambda_1/ResizeBilinear_1/sizeConst*
valueB"8  P  *
dtype0*
_output_shapes
:
Е
lambda_1/ResizeBilinear_1ResizeBilinearlambda_1/DepthToSpace_1lambda_1/ResizeBilinear_1/size*1
_output_shapes
:џџџџџџџџџИа*
align_corners( *
T0
_
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
shape: *
shared_name *
dtype0	*
_output_shapes
: *
	container 
О
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: *
use_locking(
v
Adam/iterations/readIdentityAdam/iterations*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
Z
Adam/lr/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *Зб8
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
Adam/beta_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?
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
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_2*
validate_shape(
j
Adam/beta_2/readIdentityAdam/beta_2*
T0*
_class
loc:@Adam/beta_2*
_output_shapes
: 
]
Adam/decay/initial_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
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
Adam/decayAdam/decay/initial_value*
use_locking(*
T0*
_class
loc:@Adam/decay*
validate_shape(*
_output_shapes
: 
g
Adam/decay/readIdentity
Adam/decay*
T0*
_class
loc:@Adam/decay*
_output_shapes
: 
И
lambda_1_targetPlaceholder*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*?
shape6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0
r
lambda_1_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ

loss/lambda_1_loss/subSublambda_1/ResizeBilinearlambda_1_target*1
_output_shapes
:џџџџџџџџџИа*
T0
q
loss/lambda_1_loss/AbsAbsloss/lambda_1_loss/sub*
T0*1
_output_shapes
:џџџџџџџџџИа
t
)loss/lambda_1_loss/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
З
loss/lambda_1_loss/MeanMeanloss/lambda_1_loss/Abs)loss/lambda_1_loss/Mean/reduction_indices*-
_output_shapes
:џџџџџџџџџИа*
	keep_dims( *

Tidx0*
T0
|
+loss/lambda_1_loss/Mean_1/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
В
loss/lambda_1_loss/Mean_1Meanloss/lambda_1_loss/Mean+loss/lambda_1_loss/Mean_1/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ

loss/lambda_1_loss/mulMulloss/lambda_1_loss/Mean_1lambda_1_sample_weights*#
_output_shapes
:џџџџџџџџџ*
T0
b
loss/lambda_1_loss/NotEqual/yConst*
dtype0*
_output_shapes
: *
valueB
 *    

loss/lambda_1_loss/NotEqualNotEquallambda_1_sample_weightsloss/lambda_1_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ

loss/lambda_1_loss/CastCastloss/lambda_1_loss/NotEqual*#
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0
*
Truncate( 
b
loss/lambda_1_loss/ConstConst*
_output_shapes
:*
valueB: *
dtype0

loss/lambda_1_loss/Mean_2Meanloss/lambda_1_loss/Castloss/lambda_1_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 

loss/lambda_1_loss/truedivRealDivloss/lambda_1_loss/mulloss/lambda_1_loss/Mean_2*
T0*#
_output_shapes
:џџџџџџџџџ
d
loss/lambda_1_loss/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 

loss/lambda_1_loss/Mean_3Meanloss/lambda_1_loss/truedivloss/lambda_1_loss/Const_1*
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
loss/mul/xloss/lambda_1_loss/Mean_3*
T0*
_output_shapes
: 
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
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0*
_class
loc:@loss/mul
Ї
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/lambda_1_loss/Mean_3*
_output_shapes
: *
T0*
_class
loc:@loss/mul

+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
М
Dtraining/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
dtype0

>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Dtraining/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Reshape/shape*
T0*
Tshape0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
_output_shapes
:
Ф
<training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/ShapeShapeloss/lambda_1_loss/truediv*
out_type0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
_output_shapes
:*
T0
Џ
;training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/TileTile>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Reshape<training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Shape*

Tmultiples0*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Ц
>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Shape_1Shapeloss/lambda_1_loss/truediv*
_output_shapes
:*
T0*
out_type0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3
Џ
>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Shape_2Const*
valueB *,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
dtype0*
_output_shapes
: 
Д
<training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/ConstConst*
valueB: *,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
dtype0*
_output_shapes
:
­
;training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/ProdProd>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Shape_1<training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Const*

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
_output_shapes
: 
Ж
>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Const_1Const*
valueB: *,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
dtype0*
_output_shapes
:
Б
=training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Prod_1Prod>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Shape_2>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3
А
@training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Maximum/yConst*
value	B :*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
dtype0*
_output_shapes
: 

>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/MaximumMaximum=training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Prod_1@training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Maximum/y*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
_output_shapes
: 

?training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/floordivFloorDiv;training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Prod>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Maximum*
_output_shapes
: *
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3
ђ
;training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/CastCast?training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/floordiv*

SrcT0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
Truncate( *
_output_shapes
: *

DstT0

>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/truedivRealDiv;training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Tile;training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Cast*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Т
=training/Adam/gradients/loss/lambda_1_loss/truediv_grad/ShapeShapeloss/lambda_1_loss/mul*
T0*
out_type0*-
_class#
!loc:@loss/lambda_1_loss/truediv*
_output_shapes
:
Б
?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Shape_1Const*
valueB *-
_class#
!loc:@loss/lambda_1_loss/truediv*
dtype0*
_output_shapes
: 
в
Mtraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Shape?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Shape_1*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/RealDivRealDiv>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/truedivloss/lambda_1_loss/Mean_2*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*#
_output_shapes
:џџџџџџџџџ
С
;training/Adam/gradients/loss/lambda_1_loss/truediv_grad/SumSum?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/RealDivMtraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
Б
?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/ReshapeReshape;training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Sum=training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Shape*
T0*
Tshape0*-
_class#
!loc:@loss/lambda_1_loss/truediv*#
_output_shapes
:џџџџџџџџџ
З
;training/Adam/gradients/loss/lambda_1_loss/truediv_grad/NegNegloss/lambda_1_loss/mul*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Atraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/RealDiv_1RealDiv;training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Negloss/lambda_1_loss/Mean_2*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Atraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/RealDiv_2RealDivAtraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/RealDiv_1loss/lambda_1_loss/Mean_2*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*#
_output_shapes
:џџџџџџџџџ
Ђ
;training/Adam/gradients/loss/lambda_1_loss/truediv_grad/mulMul>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/truedivAtraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/RealDiv_2*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*#
_output_shapes
:џџџџџџџџџ
С
=training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Sum_1Sum;training/Adam/gradients/loss/lambda_1_loss/truediv_grad/mulOtraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
Њ
Atraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/Reshape_1Reshape=training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Sum_1?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Shape_1*
Tshape0*-
_class#
!loc:@loss/lambda_1_loss/truediv*
_output_shapes
: *
T0
Н
9training/Adam/gradients/loss/lambda_1_loss/mul_grad/ShapeShapeloss/lambda_1_loss/Mean_1*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/lambda_1_loss/mul
Н
;training/Adam/gradients/loss/lambda_1_loss/mul_grad/Shape_1Shapelambda_1_sample_weights*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/lambda_1_loss/mul
Т
Itraining/Adam/gradients/loss/lambda_1_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/loss/lambda_1_loss/mul_grad/Shape;training/Adam/gradients/loss/lambda_1_loss/mul_grad/Shape_1*
T0*)
_class
loc:@loss/lambda_1_loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ё
7training/Adam/gradients/loss/lambda_1_loss/mul_grad/MulMul?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Reshapelambda_1_sample_weights*#
_output_shapes
:џџџџџџџџџ*
T0*)
_class
loc:@loss/lambda_1_loss/mul
­
7training/Adam/gradients/loss/lambda_1_loss/mul_grad/SumSum7training/Adam/gradients/loss/lambda_1_loss/mul_grad/MulItraining/Adam/gradients/loss/lambda_1_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*)
_class
loc:@loss/lambda_1_loss/mul
Ё
;training/Adam/gradients/loss/lambda_1_loss/mul_grad/ReshapeReshape7training/Adam/gradients/loss/lambda_1_loss/mul_grad/Sum9training/Adam/gradients/loss/lambda_1_loss/mul_grad/Shape*
Tshape0*)
_class
loc:@loss/lambda_1_loss/mul*#
_output_shapes
:џџџџџџџџџ*
T0
ѕ
9training/Adam/gradients/loss/lambda_1_loss/mul_grad/Mul_1Mulloss/lambda_1_loss/Mean_1?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Reshape*#
_output_shapes
:џџџџџџџџџ*
T0*)
_class
loc:@loss/lambda_1_loss/mul
Г
9training/Adam/gradients/loss/lambda_1_loss/mul_grad/Sum_1Sum9training/Adam/gradients/loss/lambda_1_loss/mul_grad/Mul_1Ktraining/Adam/gradients/loss/lambda_1_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*)
_class
loc:@loss/lambda_1_loss/mul*
_output_shapes
:
Ї
=training/Adam/gradients/loss/lambda_1_loss/mul_grad/Reshape_1Reshape9training/Adam/gradients/loss/lambda_1_loss/mul_grad/Sum_1;training/Adam/gradients/loss/lambda_1_loss/mul_grad/Shape_1*
T0*
Tshape0*)
_class
loc:@loss/lambda_1_loss/mul*#
_output_shapes
:џџџџџџџџџ
С
<training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/ShapeShapeloss/lambda_1_loss/Mean*
T0*
out_type0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
:
Ћ
;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/SizeConst*
_output_shapes
: *
value	B :*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
dtype0
ў
:training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/addAdd+loss/lambda_1_loss/Mean_1/reduction_indices;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Size*
_output_shapes
:*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1

:training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/modFloorMod:training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/add;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Size*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
:
Ж
>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape_1Const*
valueB:*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
dtype0*
_output_shapes
:
В
Btraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/range/startConst*
value	B : *,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
dtype0*
_output_shapes
: 
В
Btraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/range/deltaConst*
value	B :*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
dtype0*
_output_shapes
: 
р
<training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/rangeRangeBtraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/range/start;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/SizeBtraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/range/delta*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
:*

Tidx0
Б
Atraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Fill/valueConst*
value	B :*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
dtype0*
_output_shapes
: 
Ћ
;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/FillFill>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape_1Atraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Fill/value*
T0*

index_type0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
:
І
Dtraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/DynamicStitchDynamicStitch<training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/range:training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/mod<training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Fill*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
N*
_output_shapes
:
А
@training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Maximum/yConst*
value	B :*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
dtype0*
_output_shapes
: 
Є
>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/MaximumMaximumDtraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/DynamicStitch@training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Maximum/y*
_output_shapes
:*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1

?training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/floordivFloorDiv<training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Maximum*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
:
а
>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/ReshapeReshape;training/Adam/gradients/loss/lambda_1_loss/mul_grad/ReshapeDtraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/DynamicStitch*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
Tshape0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1
Ь
;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/TileTile>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Reshape?training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/floordiv*

Tmultiples0*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
У
>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape_2Shapeloss/lambda_1_loss/Mean*
T0*
out_type0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
:
Х
>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape_3Shapeloss/lambda_1_loss/Mean_1*
_output_shapes
:*
T0*
out_type0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1
Д
<training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/ConstConst*
valueB: *,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
dtype0*
_output_shapes
:
­
;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/ProdProd>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape_2<training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Const*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Ж
>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Const_1Const*
valueB: *,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
dtype0*
_output_shapes
:
Б
=training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Prod_1Prod>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape_3>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Const_1*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
В
Btraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Maximum_1/yConst*
_output_shapes
: *
value	B :*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
dtype0

@training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Maximum_1Maximum=training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Prod_1Btraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Maximum_1/y*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
: *
T0

Atraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/floordiv_1FloorDiv;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Prod@training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Maximum_1*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
: *
T0
є
;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/CastCastAtraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
Truncate( 
Љ
>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/truedivRealDiv;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Tile;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Cast*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*-
_output_shapes
:џџџџџџџџџИа
М
:training/Adam/gradients/loss/lambda_1_loss/Mean_grad/ShapeShapeloss/lambda_1_loss/Abs*
T0*
out_type0**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
:
Ї
9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/SizeConst*
_output_shapes
: *
value	B :**
_class 
loc:@loss/lambda_1_loss/Mean*
dtype0
ђ
8training/Adam/gradients/loss/lambda_1_loss/Mean_grad/addAdd)loss/lambda_1_loss/Mean/reduction_indices9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Size**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
: *
T0

8training/Adam/gradients/loss/lambda_1_loss/Mean_grad/modFloorMod8training/Adam/gradients/loss/lambda_1_loss/Mean_grad/add9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Size*
T0**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
: 
Ћ
<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape_1Const*
valueB **
_class 
loc:@loss/lambda_1_loss/Mean*
dtype0*
_output_shapes
: 
Ў
@training/Adam/gradients/loss/lambda_1_loss/Mean_grad/range/startConst*
value	B : **
_class 
loc:@loss/lambda_1_loss/Mean*
dtype0*
_output_shapes
: 
Ў
@training/Adam/gradients/loss/lambda_1_loss/Mean_grad/range/deltaConst*
value	B :**
_class 
loc:@loss/lambda_1_loss/Mean*
dtype0*
_output_shapes
: 
ж
:training/Adam/gradients/loss/lambda_1_loss/Mean_grad/rangeRange@training/Adam/gradients/loss/lambda_1_loss/Mean_grad/range/start9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Size@training/Adam/gradients/loss/lambda_1_loss/Mean_grad/range/delta**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
:*

Tidx0
­
?training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Fill/valueConst*
value	B :**
_class 
loc:@loss/lambda_1_loss/Mean*
dtype0*
_output_shapes
: 

9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/FillFill<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape_1?training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Fill/value*
T0*

index_type0**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
: 

Btraining/Adam/gradients/loss/lambda_1_loss/Mean_grad/DynamicStitchDynamicStitch:training/Adam/gradients/loss/lambda_1_loss/Mean_grad/range8training/Adam/gradients/loss/lambda_1_loss/Mean_grad/mod:training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Fill*
T0**
_class 
loc:@loss/lambda_1_loss/Mean*
N*
_output_shapes
:
Ќ
>training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :**
_class 
loc:@loss/lambda_1_loss/Mean*
dtype0

<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/MaximumMaximumBtraining/Adam/gradients/loss/lambda_1_loss/Mean_grad/DynamicStitch>training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Maximum/y**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
:*
T0

=training/Adam/gradients/loss/lambda_1_loss/Mean_grad/floordivFloorDiv:training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Maximum*
T0**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
:
к
<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/ReshapeReshape>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/truedivBtraining/Adam/gradients/loss/lambda_1_loss/Mean_grad/DynamicStitch*
T0*
Tshape0**
_class 
loc:@loss/lambda_1_loss/Mean*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
б
9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/TileTile<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Reshape=training/Adam/gradients/loss/lambda_1_loss/Mean_grad/floordiv*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*

Tmultiples0*
T0**
_class 
loc:@loss/lambda_1_loss/Mean
О
<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape_2Shapeloss/lambda_1_loss/Abs*
T0*
out_type0**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
:
П
<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape_3Shapeloss/lambda_1_loss/Mean*
T0*
out_type0**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
:
А
:training/Adam/gradients/loss/lambda_1_loss/Mean_grad/ConstConst*
valueB: **
_class 
loc:@loss/lambda_1_loss/Mean*
dtype0*
_output_shapes
:
Ѕ
9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/ProdProd<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape_2:training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
: 
В
<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Const_1Const*
valueB: **
_class 
loc:@loss/lambda_1_loss/Mean*
dtype0*
_output_shapes
:
Љ
;training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Prod_1Prod<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape_3<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/lambda_1_loss/Mean
Ў
@training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Maximum_1/yConst*
_output_shapes
: *
value	B :**
_class 
loc:@loss/lambda_1_loss/Mean*
dtype0

>training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Maximum_1Maximum;training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Prod_1@training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Maximum_1/y**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
: *
T0

?training/Adam/gradients/loss/lambda_1_loss/Mean_grad/floordiv_1FloorDiv9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Prod>training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Maximum_1*
_output_shapes
: *
T0**
_class 
loc:@loss/lambda_1_loss/Mean
ю
9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/CastCast?training/Adam/gradients/loss/lambda_1_loss/Mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0**
_class 
loc:@loss/lambda_1_loss/Mean*
Truncate( 
Ѕ
<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/truedivRealDiv9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Tile9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Cast**
_class 
loc:@loss/lambda_1_loss/Mean*1
_output_shapes
:џџџџџџџџџИа*
T0
П
8training/Adam/gradients/loss/lambda_1_loss/Abs_grad/SignSignloss/lambda_1_loss/sub*
T0*)
_class
loc:@loss/lambda_1_loss/Abs*1
_output_shapes
:џџџџџџџџџИа

7training/Adam/gradients/loss/lambda_1_loss/Abs_grad/mulMul<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/truediv8training/Adam/gradients/loss/lambda_1_loss/Abs_grad/Sign*
T0*)
_class
loc:@loss/lambda_1_loss/Abs*1
_output_shapes
:џџџџџџџџџИа
Л
9training/Adam/gradients/loss/lambda_1_loss/sub_grad/ShapeShapelambda_1/ResizeBilinear*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/lambda_1_loss/sub
Е
;training/Adam/gradients/loss/lambda_1_loss/sub_grad/Shape_1Shapelambda_1_target*
T0*
out_type0*)
_class
loc:@loss/lambda_1_loss/sub*
_output_shapes
:
Т
Itraining/Adam/gradients/loss/lambda_1_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/loss/lambda_1_loss/sub_grad/Shape;training/Adam/gradients/loss/lambda_1_loss/sub_grad/Shape_1*
T0*)
_class
loc:@loss/lambda_1_loss/sub*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
­
7training/Adam/gradients/loss/lambda_1_loss/sub_grad/SumSum7training/Adam/gradients/loss/lambda_1_loss/Abs_grad/mulItraining/Adam/gradients/loss/lambda_1_loss/sub_grad/BroadcastGradientArgs*
T0*)
_class
loc:@loss/lambda_1_loss/sub*
_output_shapes
:*

Tidx0*
	keep_dims( 
Џ
;training/Adam/gradients/loss/lambda_1_loss/sub_grad/ReshapeReshape7training/Adam/gradients/loss/lambda_1_loss/sub_grad/Sum9training/Adam/gradients/loss/lambda_1_loss/sub_grad/Shape*
Tshape0*)
_class
loc:@loss/lambda_1_loss/sub*1
_output_shapes
:џџџџџџџџџИа*
T0
Б
9training/Adam/gradients/loss/lambda_1_loss/sub_grad/Sum_1Sum7training/Adam/gradients/loss/lambda_1_loss/Abs_grad/mulKtraining/Adam/gradients/loss/lambda_1_loss/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*)
_class
loc:@loss/lambda_1_loss/sub*
_output_shapes
:
Ч
7training/Adam/gradients/loss/lambda_1_loss/sub_grad/NegNeg9training/Adam/gradients/loss/lambda_1_loss/sub_grad/Sum_1*)
_class
loc:@loss/lambda_1_loss/sub*
_output_shapes
:*
T0
Ь
=training/Adam/gradients/loss/lambda_1_loss/sub_grad/Reshape_1Reshape7training/Adam/gradients/loss/lambda_1_loss/sub_grad/Neg;training/Adam/gradients/loss/lambda_1_loss/sub_grad/Shape_1*
T0*
Tshape0*)
_class
loc:@loss/lambda_1_loss/sub*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ў
Gtraining/Adam/gradients/lambda_1/ResizeBilinear_grad/ResizeBilinearGradResizeBilinearGrad;training/Adam/gradients/loss/lambda_1_loss/sub_grad/Reshapelambda_1/DepthToSpace*
align_corners( *
T0**
_class 
loc:@lambda_1/ResizeBilinear*1
_output_shapes
:џџџџџџџџџИа
Ї
?training/Adam/gradients/lambda_1/DepthToSpace_grad/SpaceToDepthSpaceToDepthGtraining/Adam/gradients/lambda_1/ResizeBilinear_grad/ResizeBilinearGrad*

block_size*
T0*(
_class
loc:@lambda_1/DepthToSpace*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџЈ
ъ
9training/Adam/gradients/conv2d_7/BiasAdd_grad/BiasAddGradBiasAddGrad?training/Adam/gradients/lambda_1/DepthToSpace_grad/SpaceToDepth*
T0*#
_class
loc:@conv2d_7/BiasAdd*
data_formatNHWC*
_output_shapes
:
л
8training/Adam/gradients/conv2d_7/convolution_grad/ShapeNShapeNdropout_3/cond/Mergeconv2d_7/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_7/convolution*
N* 
_output_shapes
::
Р
Etraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_7/convolution_grad/ShapeNconv2d_7/kernel/read?training/Adam/gradients/lambda_1/DepthToSpace_grad/SpaceToDepth*
	dilations
*
T0*'
_class
loc:@conv2d_7/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ@
Й
Ftraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdropout_3/cond/Merge:training/Adam/gradients/conv2d_7/convolution_grad/ShapeN:1?training/Adam/gradients/lambda_1/DepthToSpace_grad/SpaceToDepth*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@*
	dilations
*
T0*'
_class
loc:@conv2d_7/convolution*
data_formatNHWC*
strides

І
;training/Adam/gradients/dropout_3/cond/Merge_grad/cond_gradSwitchEtraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropInputdropout_3/cond/pred_id*'
_class
loc:@conv2d_7/convolution*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@*
T0
о
training/Adam/gradients/SwitchSwitchleaky_re_lu_6/LeakyReludropout_3/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@
Ж
 training/Adam/gradients/IdentityIdentity training/Adam/gradients/Switch:1*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*1
_output_shapes
:џџџџџџџџџЈ@
Ћ
training/Adam/gradients/Shape_1Shape training/Adam/gradients/Switch:1*
out_type0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*
_output_shapes
:*
T0
З
#training/Adam/gradients/zeros/ConstConst!^training/Adam/gradients/Identity*
valueB
 *    **
_class 
loc:@leaky_re_lu_6/LeakyRelu*
dtype0*
_output_shapes
: 
х
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*
T0*

index_type0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*1
_output_shapes
:џџџџџџџџџЈ@

>training/Adam/gradients/dropout_3/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_3/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros*
N*3
_output_shapes!
:џџџџџџџџџЈ@: *
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu
Ъ
=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/ShapeShapedropout_3/cond/dropout/truediv*
T0*
out_type0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
_output_shapes
:
Ъ
?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape_1Shapedropout_3/cond/dropout/Floor*
T0*
out_type0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
_output_shapes
:
в
Mtraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul

;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_3/cond/Merge_grad/cond_grad:1dropout_3/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџЈ@
Н
;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul
П
?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape*
T0*
Tshape0*-
_class#
!loc:@dropout_3/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџЈ@

=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Mul_1Muldropout_3/cond/dropout/truediv=training/Adam/gradients/dropout_3/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџЈ@
У
=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
Х
Atraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*-
_class#
!loc:@dropout_3/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџЈ@
Ц
Atraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/ShapeShapedropout_3/cond/mul*
T0*
out_type0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*
_output_shapes
:
Й
Ctraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Shape_1Const*
valueB *1
_class'
%#loc:@dropout_3/cond/dropout/truediv*
dtype0*
_output_shapes
: 
т
Qtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsAtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/ShapeCtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv

Ctraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/RealDivRealDiv?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Reshapedropout_3/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@
б
?training/Adam/gradients/dropout_3/cond/dropout/truediv_grad/SumSumCtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/RealDivQtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*
_output_shapes
:
Я
Ctraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/ReshapeReshape?training/Adam/gradients/dropout_3/cond/dropout/truediv_grad/SumAtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Shape*
T0*
Tshape0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@
Щ
?training/Adam/gradients/dropout_3/cond/dropout/truediv_grad/NegNegdropout_3/cond/mul*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@

Etraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/RealDiv_1RealDiv?training/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Negdropout_3/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@
Ђ
Etraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/RealDiv_2RealDivEtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/RealDiv_1dropout_3/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџЈ@*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv
Н
?training/Adam/gradients/dropout_3/cond/dropout/truediv_grad/mulMul?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/ReshapeEtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/RealDiv_2*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@
б
Atraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Sum_1Sum?training/Adam/gradients/dropout_3/cond/dropout/truediv_grad/mulStraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
К
Etraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Reshape_1ReshapeAtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Sum_1Ctraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Shape_1*
Tshape0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*
_output_shapes
: *
T0
З
5training/Adam/gradients/dropout_3/cond/mul_grad/ShapeShapedropout_3/cond/mul/Switch:1*
_output_shapes
:*
T0*
out_type0*%
_class
loc:@dropout_3/cond/mul
Ё
7training/Adam/gradients/dropout_3/cond/mul_grad/Shape_1Const*
valueB *%
_class
loc:@dropout_3/cond/mul*
dtype0*
_output_shapes
: 
В
Etraining/Adam/gradients/dropout_3/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_3/cond/mul_grad/Shape7training/Adam/gradients/dropout_3/cond/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@dropout_3/cond/mul
ј
3training/Adam/gradients/dropout_3/cond/mul_grad/MulMulCtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Reshapedropout_3/cond/mul/y*
T0*%
_class
loc:@dropout_3/cond/mul*1
_output_shapes
:џџџџџџџџџЈ@

3training/Adam/gradients/dropout_3/cond/mul_grad/SumSum3training/Adam/gradients/dropout_3/cond/mul_grad/MulEtraining/Adam/gradients/dropout_3/cond/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*%
_class
loc:@dropout_3/cond/mul*
_output_shapes
:

7training/Adam/gradients/dropout_3/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_3/cond/mul_grad/Sum5training/Adam/gradients/dropout_3/cond/mul_grad/Shape*
T0*
Tshape0*%
_class
loc:@dropout_3/cond/mul*1
_output_shapes
:џџџџџџџџџЈ@

5training/Adam/gradients/dropout_3/cond/mul_grad/Mul_1Muldropout_3/cond/mul/Switch:1Ctraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Reshape*1
_output_shapes
:џџџџџџџџџЈ@*
T0*%
_class
loc:@dropout_3/cond/mul
Ѓ
5training/Adam/gradients/dropout_3/cond/mul_grad/Sum_1Sum5training/Adam/gradients/dropout_3/cond/mul_grad/Mul_1Gtraining/Adam/gradients/dropout_3/cond/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*%
_class
loc:@dropout_3/cond/mul*
_output_shapes
:

9training/Adam/gradients/dropout_3/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_3/cond/mul_grad/Sum_17training/Adam/gradients/dropout_3/cond/mul_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*%
_class
loc:@dropout_3/cond/mul
р
 training/Adam/gradients/Switch_1Switchleaky_re_lu_6/LeakyReludropout_3/cond/pred_id*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu
И
"training/Adam/gradients/Identity_1Identity training/Adam/gradients/Switch_1*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*1
_output_shapes
:џџџџџџџџџЈ@
Ћ
training/Adam/gradients/Shape_2Shape training/Adam/gradients/Switch_1*
out_type0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*
_output_shapes
:*
T0
Л
%training/Adam/gradients/zeros_1/ConstConst#^training/Adam/gradients/Identity_1*
_output_shapes
: *
valueB
 *    **
_class 
loc:@leaky_re_lu_6/LeakyRelu*
dtype0
щ
training/Adam/gradients/zeros_1Filltraining/Adam/gradients/Shape_2%training/Adam/gradients/zeros_1/Const*
T0*

index_type0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*1
_output_shapes
:џџџџџџџџџЈ@

@training/Adam/gradients/dropout_3/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_17training/Adam/gradients/dropout_3/cond/mul_grad/Reshape*
N*3
_output_shapes!
:џџџџџџџџџЈ@: *
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu

training/Adam/gradients/AddNAddN>training/Adam/gradients/dropout_3/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_3/cond/mul/Switch_grad/cond_grad*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*
N*1
_output_shapes
:џџџџџџџџџЈ@
ћ
Btraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGradLeakyReluGradtraining/Adam/gradients/AddNconv2d_6/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*
alpha%>*1
_output_shapes
:џџџџџџџџџЈ@
э
9training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_6/BiasAdd*
data_formatNHWC*
_output_shapes
:@
о
8training/Adam/gradients/conv2d_6/convolution_grad/ShapeNShapeNleaky_re_lu_5/LeakyReluconv2d_6/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_6/convolution*
N* 
_output_shapes
::
У
Etraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_6/convolution_grad/ShapeNconv2d_6/kernel/readBtraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGrad*'
_class
loc:@conv2d_6/convolution*
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
П
Ftraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterleaky_re_lu_5/LeakyRelu:training/Adam/gradients/conv2d_6/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@@*
	dilations
*
T0*'
_class
loc:@conv2d_6/convolution
Є
Btraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGradLeakyReluGradEtraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputconv2d_5/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_5/LeakyRelu*
alpha%>*1
_output_shapes
:џџџџџџџџџЈ@
э
9training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGrad*#
_class
loc:@conv2d_5/BiasAdd*
data_formatNHWC*
_output_shapes
:@*
T0
ь
8training/Adam/gradients/conv2d_5/convolution_grad/ShapeNShapeN%up_sampling2d_1/ResizeNearestNeighborconv2d_5/kernel/read* 
_output_shapes
::*
T0*
out_type0*'
_class
loc:@conv2d_5/convolution*
N
Ф
Etraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_5/convolution_grad/ShapeNconv2d_5/kernel/readBtraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGrad*
paddingSAME*2
_output_shapes 
:џџџџџџџџџЈ*
	dilations
*
T0*'
_class
loc:@conv2d_5/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ю
Ftraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_1/ResizeNearestNeighbor:training/Adam/gradients/conv2d_5/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGrad*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*'
_class
loc:@conv2d_5/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
ь
atraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"    *8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*
dtype0*
_output_shapes
:
Џ
\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradEtraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropInputatraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*2
_output_shapes 
:џџџџџџџџџ*
align_corners( *
T0

@training/Adam/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGraddropout_2/cond/Mergemax_pooling2d_2/MaxPool\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*2
_output_shapes 
:џџџџџџџџџЈ*
T0**
_class 
loc:@max_pooling2d_2/MaxPool
І
;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradSwitch@training/Adam/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGraddropout_2/cond/pred_id*P
_output_shapes>
<:џџџџџџџџџЈ:џџџџџџџџџЈ*
T0**
_class 
loc:@max_pooling2d_2/MaxPool
т
 training/Adam/gradients/Switch_2Switchleaky_re_lu_4/LeakyReludropout_2/cond/pred_id*P
_output_shapes>
<:џџџџџџџџџЈ:џџџџџџџџџЈ*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu
Л
"training/Adam/gradients/Identity_2Identity"training/Adam/gradients/Switch_2:1*2
_output_shapes 
:џџџџџџџџџЈ*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu
­
training/Adam/gradients/Shape_3Shape"training/Adam/gradients/Switch_2:1*
T0*
out_type0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*
_output_shapes
:
Л
%training/Adam/gradients/zeros_2/ConstConst#^training/Adam/gradients/Identity_2*
valueB
 *    **
_class 
loc:@leaky_re_lu_4/LeakyRelu*
dtype0*
_output_shapes
: 
ъ
training/Adam/gradients/zeros_2Filltraining/Adam/gradients/Shape_3%training/Adam/gradients/zeros_2/Const*
T0*

index_type0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*2
_output_shapes 
:џџџџџџџџџЈ

>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_2**
_class 
loc:@leaky_re_lu_4/LeakyRelu*
N*4
_output_shapes"
 :џџџџџџџџџЈ: *
T0
Ъ
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ShapeShapedropout_2/cond/dropout/truediv*
_output_shapes
:*
T0*
out_type0*-
_class#
!loc:@dropout_2/cond/dropout/mul
Ъ
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1Shapedropout_2/cond/dropout/Floor*
T0*
out_type0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:
в
Mtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1dropout_2/cond/dropout/Floor*2
_output_shapes 
:џџџџџџџџџЈ*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
Н
;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
Р
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape*
T0*
Tshape0*-
_class#
!loc:@dropout_2/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџЈ

=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Muldropout_2/cond/dropout/truediv=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџЈ
У
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ц
Atraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
Tshape0*-
_class#
!loc:@dropout_2/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџЈ*
T0
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
Qtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsAtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/ShapeCtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv

Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDivRealDiv?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshapedropout_2/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџЈ
б
?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/SumSumCtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDivQtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv
а
Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/ReshapeReshape?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/SumAtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Shape*
T0*
Tshape0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџЈ
Ъ
?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/NegNegdropout_2/cond/mul*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџЈ

Etraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_1RealDiv?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Negdropout_2/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџЈ
Ѓ
Etraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_2RealDivEtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_1dropout_2/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџЈ
О
?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/mulMul?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeEtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_2*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџЈ
б
Atraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Sum_1Sum?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/mulStraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv
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
Etraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_2/cond/mul_grad/Shape7training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_2/cond/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
љ
3training/Adam/gradients/dropout_2/cond/mul_grad/MulMulCtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Reshapedropout_2/cond/mul/y*2
_output_shapes 
:џџџџџџџџџЈ*
T0*%
_class
loc:@dropout_2/cond/mul

3training/Adam/gradients/dropout_2/cond/mul_grad/SumSum3training/Adam/gradients/dropout_2/cond/mul_grad/MulEtraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs*
T0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
 
7training/Adam/gradients/dropout_2/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_2/cond/mul_grad/Sum5training/Adam/gradients/dropout_2/cond/mul_grad/Shape*
T0*
Tshape0*%
_class
loc:@dropout_2/cond/mul*2
_output_shapes 
:џџџџџџџџџЈ

5training/Adam/gradients/dropout_2/cond/mul_grad/Mul_1Muldropout_2/cond/mul/Switch:1Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Reshape*%
_class
loc:@dropout_2/cond/mul*2
_output_shapes 
:џџџџџџџџџЈ*
T0
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
т
 training/Adam/gradients/Switch_3Switchleaky_re_lu_4/LeakyReludropout_2/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*P
_output_shapes>
<:џџџџџџџџџЈ:џџџџџџџџџЈ
Й
"training/Adam/gradients/Identity_3Identity training/Adam/gradients/Switch_3*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*2
_output_shapes 
:џџџџџџџџџЈ
Ћ
training/Adam/gradients/Shape_4Shape training/Adam/gradients/Switch_3*
T0*
out_type0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*
_output_shapes
:
Л
%training/Adam/gradients/zeros_3/ConstConst#^training/Adam/gradients/Identity_3*
dtype0*
_output_shapes
: *
valueB
 *    **
_class 
loc:@leaky_re_lu_4/LeakyRelu
ъ
training/Adam/gradients/zeros_3Filltraining/Adam/gradients/Shape_4%training/Adam/gradients/zeros_3/Const*
T0*

index_type0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*2
_output_shapes 
:џџџџџџџџџЈ

@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_37training/Adam/gradients/dropout_2/cond/mul_grad/Reshape*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*
N*4
_output_shapes"
 :џџџџџџџџџЈ: 

training/Adam/gradients/AddN_1AddN>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_grad*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*
N*2
_output_shapes 
:џџџџџџџџџЈ
ў
Btraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGradLeakyReluGradtraining/Adam/gradients/AddN_1conv2d_4/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*
alpha%>*2
_output_shapes 
:џџџџџџџџџЈ
ю
9training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGrad*#
_class
loc:@conv2d_4/BiasAdd*
data_formatNHWC*
_output_shapes	
:*
T0
о
8training/Adam/gradients/conv2d_4/convolution_grad/ShapeNShapeNleaky_re_lu_3/LeakyReluconv2d_4/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_4/convolution*
N* 
_output_shapes
::
Ф
Etraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_4/convolution_grad/ShapeNconv2d_4/kernel/readBtraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_4/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:џџџџџџџџџЈ
С
Ftraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterleaky_re_lu_3/LeakyRelu:training/Adam/gradients/conv2d_4/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGrad*(
_output_shapes
:*
	dilations
*
T0*'
_class
loc:@conv2d_4/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ѕ
Btraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGradLeakyReluGradEtraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputconv2d_3/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_3/LeakyRelu*
alpha%>*2
_output_shapes 
:џџџџџџџџџЈ
ю
9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGrad*#
_class
loc:@conv2d_3/BiasAdd*
data_formatNHWC*
_output_shapes	
:*
T0
о
8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNShapeNmax_pooling2d_1/MaxPoolconv2d_3/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_3/convolution*
N* 
_output_shapes
::
У
Etraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNconv2d_3/kernel/readBtraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGrad*1
_output_shapes
:џџџџџџџџџЈ@*
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
Р
Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_1/MaxPool:training/Adam/gradients/conv2d_3/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_3/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@
џ
@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGraddropout_1/cond/Mergemax_pooling2d_1/MaxPoolEtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInput*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа@*
T0**
_class 
loc:@max_pooling2d_1/MaxPool*
strides
*
data_formatNHWC
Є
;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradSwitch@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGraddropout_1/cond/pred_id*N
_output_shapes<
::џџџџџџџџџИа@:џџџџџџџџџИа@*
T0**
_class 
loc:@max_pooling2d_1/MaxPool
р
 training/Adam/gradients/Switch_4Switchleaky_re_lu_2/LeakyReludropout_1/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*N
_output_shapes<
::џџџџџџџџџИа@:џџџџџџџџџИа@
К
"training/Adam/gradients/Identity_4Identity"training/Adam/gradients/Switch_4:1*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*1
_output_shapes
:џџџџџџџџџИа@
­
training/Adam/gradients/Shape_5Shape"training/Adam/gradients/Switch_4:1*
out_type0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
_output_shapes
:*
T0
Л
%training/Adam/gradients/zeros_4/ConstConst#^training/Adam/gradients/Identity_4*
_output_shapes
: *
valueB
 *    **
_class 
loc:@leaky_re_lu_2/LeakyRelu*
dtype0
щ
training/Adam/gradients/zeros_4Filltraining/Adam/gradients/Shape_5%training/Adam/gradients/zeros_4/Const*
T0*

index_type0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*1
_output_shapes
:џџџџџџџџџИа@

>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_4*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџИа@: 
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
Mtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1dropout_1/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџИа@
Н
;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
П
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape*
Tshape0*-
_class#
!loc:@dropout_1/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџИа@*
T0

=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Muldropout_1/cond/dropout/truediv=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџИа@
У
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:
Х
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*1
_output_shapes
:џџџџџџџџџИа@*
T0*
Tshape0*-
_class#
!loc:@dropout_1/cond/dropout/mul
Ц
Atraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/ShapeShapedropout_1/cond/mul*
_output_shapes
:*
T0*
out_type0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
Й
Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1Const*
_output_shapes
: *
valueB *1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
dtype0
т
Qtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsAtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/ShapeCtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv

Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDivRealDiv?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshapedropout_1/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа@
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
Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/ReshapeReshape?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/SumAtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Shape*
T0*
Tshape0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа@
Щ
?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/NegNegdropout_1/cond/mul*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа@

Etraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1RealDiv?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Negdropout_1/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа@
Ђ
Etraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2RealDivEtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1dropout_1/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа@
Н
?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/mulMul?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeEtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2*1
_output_shapes
:џџџџџџџџџИа@*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
б
Atraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1Sum?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/mulStraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgs:1*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
К
Etraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Reshape_1ReshapeAtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1*
T0*
Tshape0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
: 
З
5training/Adam/gradients/dropout_1/cond/mul_grad/ShapeShapedropout_1/cond/mul/Switch:1*
T0*
out_type0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:
Ё
7training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1Const*
valueB *%
_class
loc:@dropout_1/cond/mul*
dtype0*
_output_shapes
: 
В
Etraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_1/cond/mul_grad/Shape7training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_1/cond/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ј
3training/Adam/gradients/dropout_1/cond/mul_grad/MulMulCtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Reshapedropout_1/cond/mul/y*%
_class
loc:@dropout_1/cond/mul*1
_output_shapes
:џџџџџџџџџИа@*
T0

3training/Adam/gradients/dropout_1/cond/mul_grad/SumSum3training/Adam/gradients/dropout_1/cond/mul_grad/MulEtraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*%
_class
loc:@dropout_1/cond/mul

7training/Adam/gradients/dropout_1/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_1/cond/mul_grad/Sum5training/Adam/gradients/dropout_1/cond/mul_grad/Shape*
Tshape0*%
_class
loc:@dropout_1/cond/mul*1
_output_shapes
:џџџџџџџџџИа@*
T0

5training/Adam/gradients/dropout_1/cond/mul_grad/Mul_1Muldropout_1/cond/mul/Switch:1Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Reshape*
T0*%
_class
loc:@dropout_1/cond/mul*1
_output_shapes
:џџџџџџџџџИа@
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
9training/Adam/gradients/dropout_1/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_1/cond/mul_grad/Sum_17training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1*
T0*
Tshape0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
: 
р
 training/Adam/gradients/Switch_5Switchleaky_re_lu_2/LeakyReludropout_1/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*N
_output_shapes<
::џџџџџџџџџИа@:џџџџџџџџџИа@
И
"training/Adam/gradients/Identity_5Identity training/Adam/gradients/Switch_5*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*1
_output_shapes
:џџџџџџџџџИа@
Ћ
training/Adam/gradients/Shape_6Shape training/Adam/gradients/Switch_5*
T0*
out_type0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
_output_shapes
:
Л
%training/Adam/gradients/zeros_5/ConstConst#^training/Adam/gradients/Identity_5*
dtype0*
_output_shapes
: *
valueB
 *    **
_class 
loc:@leaky_re_lu_2/LeakyRelu
щ
training/Adam/gradients/zeros_5Filltraining/Adam/gradients/Shape_6%training/Adam/gradients/zeros_5/Const*

index_type0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*1
_output_shapes
:џџџџџџџџџИа@*
T0

@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_57training/Adam/gradients/dropout_1/cond/mul_grad/Reshape*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџИа@: 

training/Adam/gradients/AddN_2AddN>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_grad*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N*1
_output_shapes
:џџџџџџџџџИа@
§
Btraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGradLeakyReluGradtraining/Adam/gradients/AddN_2conv2d_2/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
alpha%>*1
_output_shapes
:џџџџџџџџџИа@
э
9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
_output_shapes
:@*
T0*#
_class
loc:@conv2d_2/BiasAdd*
data_formatNHWC
о
8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNShapeNleaky_re_lu_1/LeakyReluconv2d_2/kernel/read* 
_output_shapes
::*
T0*
out_type0*'
_class
loc:@conv2d_2/convolution*
N
У
Etraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNconv2d_2/kernel/readBtraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*1
_output_shapes
:џџџџџџџџџИа@*
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
П
Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterleaky_re_lu_1/LeakyRelu:training/Adam/gradients/conv2d_2/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@@
Є
Btraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGradLeakyReluGradEtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputconv2d_1/BiasAdd**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
alpha%>*1
_output_shapes
:џџџџџџџџџИа@*
T0
э
9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*#
_class
loc:@conv2d_1/BiasAdd*
data_formatNHWC*
_output_shapes
:@*
T0
е
8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNShapeNconv2d_1_inputconv2d_1/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_1/convolution*
N* 
_output_shapes
::
У
Etraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNconv2d_1/kernel/readBtraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*1
_output_shapes
:џџџџџџџџџИа*
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
Ж
Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_1_input:training/Adam/gradients/conv2d_1/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
T0*'
_class
loc:@conv2d_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@*
	dilations

_
training/Adam/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Ќ
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*"
_class
loc:@Adam/iterations*
_output_shapes
: *
use_locking( *
T0	
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
training/Adam/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
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
training/Adam/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
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
|
#training/Adam/zeros/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
^
training/Adam/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zerosFill#training/Adam/zeros/shape_as_tensortraining/Adam/zeros/Const*
T0*

index_type0*&
_output_shapes
:@

training/Adam/Variable
VariableV2*
dtype0*&
_output_shapes
:@*
	container *
shape:@*
shared_name 
й
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable

training/Adam/Variable/readIdentitytraining/Adam/Variable*
T0*)
_class
loc:@training/Adam/Variable*&
_output_shapes
:@
b
training/Adam/zeros_1Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_1
VariableV2*
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
е
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:@
~
%training/Adam/zeros_2/shape_as_tensorConst*
_output_shapes
:*%
valueB"      @   @   *
dtype0
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
:@@

training/Adam/Variable_2
VariableV2*
shared_name *
dtype0*&
_output_shapes
:@@*
	container *
shape:@@
с
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*&
_output_shapes
:@@
Ё
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
T0*+
_class!
loc:@training/Adam/Variable_2*&
_output_shapes
:@@
b
training/Adam/zeros_3Const*
_output_shapes
:@*
valueB@*    *
dtype0

training/Adam/Variable_3
VariableV2*
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
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
valueB"      @      *
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
Ѕ
training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*'
_output_shapes
:@*
T0*

index_type0

training/Adam/Variable_4
VariableV2*
dtype0*'
_output_shapes
:@*
	container *
shape:@*
shared_name 
т
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*'
_output_shapes
:@
Ђ
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4*'
_output_shapes
:@
d
training/Adam/zeros_5Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_5
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
ж
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes	
:
~
%training/Adam/zeros_6/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
`
training/Adam/zeros_6/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
І
training/Adam/zeros_6Fill%training/Adam/zeros_6/shape_as_tensortraining/Adam/zeros_6/Const*
T0*

index_type0*(
_output_shapes
:
 
training/Adam/Variable_6
VariableV2*(
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
у
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*(
_output_shapes
:
Ѓ
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
T0*+
_class!
loc:@training/Adam/Variable_6*(
_output_shapes
:
d
training/Adam/zeros_7Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_7
VariableV2*
_output_shapes	
:*
	container *
shape:*
shared_name *
dtype0
ж
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes	
:*
T0
~
%training/Adam/zeros_8/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"         @   
`
training/Adam/zeros_8/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ѕ
training/Adam/zeros_8Fill%training/Adam/zeros_8/shape_as_tensortraining/Adam/zeros_8/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_8
VariableV2*'
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
т
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*'
_output_shapes
:@
Ђ
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
T0*+
_class!
loc:@training/Adam/Variable_8*'
_output_shapes
:@
b
training/Adam/zeros_9Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_9
VariableV2*
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
е
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
:@

&training/Adam/zeros_10/shape_as_tensorConst*%
valueB"      @   @   *
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
:@@

training/Adam/Variable_10
VariableV2*&
_output_shapes
:@@*
	container *
shape:@@*
shared_name *
dtype0
х
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*&
_output_shapes
:@@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(
Є
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*&
_output_shapes
:@@
c
training/Adam/zeros_11Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_11
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
й
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11

training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
:@
{
training/Adam/zeros_12Const*%
valueB@*    *
dtype0*&
_output_shapes
:@

training/Adam/Variable_12
VariableV2*
dtype0*&
_output_shapes
:@*
	container *
shape:@*
shared_name 
х
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*&
_output_shapes
:@
Є
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*&
_output_shapes
:@
c
training/Adam/zeros_13Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_13
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:

training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
T0*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
:

&training/Adam/zeros_14/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_14/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ї
training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*
T0*

index_type0*&
_output_shapes
:@

training/Adam/Variable_14
VariableV2*
shape:@*
shared_name *
dtype0*&
_output_shapes
:@*
	container 
х
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14
Є
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*&
_output_shapes
:@
c
training/Adam/zeros_15Const*
_output_shapes
:@*
valueB@*    *
dtype0

training/Adam/Variable_15
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
й
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
:@

&training/Adam/zeros_16/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
a
training/Adam/zeros_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ї
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*
T0*

index_type0*&
_output_shapes
:@@

training/Adam/Variable_16
VariableV2*
shared_name *
dtype0*&
_output_shapes
:@@*
	container *
shape:@@
х
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*&
_output_shapes
:@@
Є
training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*
T0*,
_class"
 loc:@training/Adam/Variable_16*&
_output_shapes
:@@
c
training/Adam/zeros_17Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_17
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
й
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
_output_shapes
:@

&training/Adam/zeros_18/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @      
a
training/Adam/zeros_18/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ј
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_18
VariableV2*
shared_name *
dtype0*'
_output_shapes
:@*
	container *
shape:@
ц
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*'
_output_shapes
:@
Ѕ
training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*'
_output_shapes
:@*
T0
e
training/Adam/zeros_19Const*
_output_shapes	
:*
valueB*    *
dtype0

training/Adam/Variable_19
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
к
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes	
:*
use_locking(

training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
_output_shapes	
:

&training/Adam/zeros_20/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_20/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Љ
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*
T0*

index_type0*(
_output_shapes
:
Ё
training/Adam/Variable_20
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ч
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(*(
_output_shapes
:
І
training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_20
e
training/Adam/zeros_21Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_21
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
к
 training/Adam/Variable_21/AssignAssigntraining/Adam/Variable_21training/Adam/zeros_21*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_21

&training/Adam/zeros_22/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_22/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ј
training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_22
VariableV2*
dtype0*'
_output_shapes
:@*
	container *
shape:@*
shared_name 
ц
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22
Ѕ
training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*'
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_22
c
training/Adam/zeros_23Const*
dtype0*
_output_shapes
:@*
valueB@*    

training/Adam/Variable_23
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
й
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23

training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_23

&training/Adam/zeros_24/shape_as_tensorConst*
_output_shapes
:*%
valueB"      @   @   *
dtype0
a
training/Adam/zeros_24/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ї
training/Adam/zeros_24Fill&training/Adam/zeros_24/shape_as_tensortraining/Adam/zeros_24/Const*&
_output_shapes
:@@*
T0*

index_type0

training/Adam/Variable_24
VariableV2*
dtype0*&
_output_shapes
:@@*
	container *
shape:@@*
shared_name 
х
 training/Adam/Variable_24/AssignAssigntraining/Adam/Variable_24training/Adam/zeros_24*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(*&
_output_shapes
:@@
Є
training/Adam/Variable_24/readIdentitytraining/Adam/Variable_24*
T0*,
_class"
 loc:@training/Adam/Variable_24*&
_output_shapes
:@@
c
training/Adam/zeros_25Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_25
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
й
 training/Adam/Variable_25/AssignAssigntraining/Adam/Variable_25training/Adam/zeros_25*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25

training/Adam/Variable_25/readIdentitytraining/Adam/Variable_25*
T0*,
_class"
 loc:@training/Adam/Variable_25*
_output_shapes
:@
{
training/Adam/zeros_26Const*&
_output_shapes
:@*%
valueB@*    *
dtype0

training/Adam/Variable_26
VariableV2*
dtype0*&
_output_shapes
:@*
	container *
shape:@*
shared_name 
х
 training/Adam/Variable_26/AssignAssigntraining/Adam/Variable_26training/Adam/zeros_26*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_26*
validate_shape(*&
_output_shapes
:@
Є
training/Adam/Variable_26/readIdentitytraining/Adam/Variable_26*
T0*,
_class"
 loc:@training/Adam/Variable_26*&
_output_shapes
:@
c
training/Adam/zeros_27Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_27
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_27/AssignAssigntraining/Adam/Variable_27training/Adam/zeros_27*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_27*
validate_shape(*
_output_shapes
:

training/Adam/Variable_27/readIdentitytraining/Adam/Variable_27*
T0*,
_class"
 loc:@training/Adam/Variable_27*
_output_shapes
:
p
&training/Adam/zeros_28/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_28/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_28Fill&training/Adam/zeros_28/shape_as_tensortraining/Adam/zeros_28/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_28
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_28/AssignAssigntraining/Adam/Variable_28training/Adam/zeros_28*
T0*,
_class"
 loc:@training/Adam/Variable_28*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_28/readIdentitytraining/Adam/Variable_28*
T0*,
_class"
 loc:@training/Adam/Variable_28*
_output_shapes
:
p
&training/Adam/zeros_29/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_29/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_29Fill&training/Adam/zeros_29/shape_as_tensortraining/Adam/zeros_29/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_29
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
й
 training/Adam/Variable_29/AssignAssigntraining/Adam/Variable_29training/Adam/zeros_29*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_29*
validate_shape(*
_output_shapes
:

training/Adam/Variable_29/readIdentitytraining/Adam/Variable_29*
T0*,
_class"
 loc:@training/Adam/Variable_29*
_output_shapes
:
p
&training/Adam/zeros_30/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_30/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_30Fill&training/Adam/zeros_30/shape_as_tensortraining/Adam/zeros_30/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_30
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_30/AssignAssigntraining/Adam/Variable_30training/Adam/zeros_30*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_30

training/Adam/Variable_30/readIdentitytraining/Adam/Variable_30*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_30
p
&training/Adam/zeros_31/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_31/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_31Fill&training/Adam/zeros_31/shape_as_tensortraining/Adam/zeros_31/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_31
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_31/AssignAssigntraining/Adam/Variable_31training/Adam/zeros_31*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_31*
validate_shape(*
_output_shapes
:

training/Adam/Variable_31/readIdentitytraining/Adam/Variable_31*
T0*,
_class"
 loc:@training/Adam/Variable_31*
_output_shapes
:
p
&training/Adam/zeros_32/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_32/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_32Fill&training/Adam/zeros_32/shape_as_tensortraining/Adam/zeros_32/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_32
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_32/AssignAssigntraining/Adam/Variable_32training/Adam/zeros_32*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_32

training/Adam/Variable_32/readIdentitytraining/Adam/Variable_32*
T0*,
_class"
 loc:@training/Adam/Variable_32*
_output_shapes
:
p
&training/Adam/zeros_33/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_33/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_33Fill&training/Adam/zeros_33/shape_as_tensortraining/Adam/zeros_33/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_33
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_33/AssignAssigntraining/Adam/Variable_33training/Adam/zeros_33*,
_class"
 loc:@training/Adam/Variable_33*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_33/readIdentitytraining/Adam/Variable_33*
T0*,
_class"
 loc:@training/Adam/Variable_33*
_output_shapes
:
p
&training/Adam/zeros_34/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_34/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_34Fill&training/Adam/zeros_34/shape_as_tensortraining/Adam/zeros_34/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_34
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_34/AssignAssigntraining/Adam/Variable_34training/Adam/zeros_34*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_34*
validate_shape(*
_output_shapes
:

training/Adam/Variable_34/readIdentitytraining/Adam/Variable_34*
T0*,
_class"
 loc:@training/Adam/Variable_34*
_output_shapes
:
p
&training/Adam/zeros_35/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_35/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_35Fill&training/Adam/zeros_35/shape_as_tensortraining/Adam/zeros_35/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_35
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_35/AssignAssigntraining/Adam/Variable_35training/Adam/zeros_35*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_35

training/Adam/Variable_35/readIdentitytraining/Adam/Variable_35*
T0*,
_class"
 loc:@training/Adam/Variable_35*
_output_shapes
:
p
&training/Adam/zeros_36/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_36/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_36Fill&training/Adam/zeros_36/shape_as_tensortraining/Adam/zeros_36/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_36
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_36/AssignAssigntraining/Adam/Variable_36training/Adam/zeros_36*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_36*
validate_shape(*
_output_shapes
:

training/Adam/Variable_36/readIdentitytraining/Adam/Variable_36*
T0*,
_class"
 loc:@training/Adam/Variable_36*
_output_shapes
:
p
&training/Adam/zeros_37/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_37/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_37Fill&training/Adam/zeros_37/shape_as_tensortraining/Adam/zeros_37/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_37
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_37/AssignAssigntraining/Adam/Variable_37training/Adam/zeros_37*
T0*,
_class"
 loc:@training/Adam/Variable_37*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_37/readIdentitytraining/Adam/Variable_37*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_37
p
&training/Adam/zeros_38/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_38/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_38Fill&training/Adam/zeros_38/shape_as_tensortraining/Adam/zeros_38/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_38
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_38/AssignAssigntraining/Adam/Variable_38training/Adam/zeros_38*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_38*
validate_shape(*
_output_shapes
:

training/Adam/Variable_38/readIdentitytraining/Adam/Variable_38*
T0*,
_class"
 loc:@training/Adam/Variable_38*
_output_shapes
:
p
&training/Adam/zeros_39/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_39/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_39Fill&training/Adam/zeros_39/shape_as_tensortraining/Adam/zeros_39/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_39
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
й
 training/Adam/Variable_39/AssignAssigntraining/Adam/Variable_39training/Adam/zeros_39*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_39*
validate_shape(*
_output_shapes
:

training/Adam/Variable_39/readIdentitytraining/Adam/Variable_39*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_39
p
&training/Adam/zeros_40/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_40/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_40Fill&training/Adam/zeros_40/shape_as_tensortraining/Adam/zeros_40/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_40
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_40/AssignAssigntraining/Adam/Variable_40training/Adam/zeros_40*,
_class"
 loc:@training/Adam/Variable_40*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_40/readIdentitytraining/Adam/Variable_40*
T0*,
_class"
 loc:@training/Adam/Variable_40*
_output_shapes
:
p
&training/Adam/zeros_41/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_41/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_41Fill&training/Adam/zeros_41/shape_as_tensortraining/Adam/zeros_41/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_41
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
й
 training/Adam/Variable_41/AssignAssigntraining/Adam/Variable_41training/Adam/zeros_41*
T0*,
_class"
 loc:@training/Adam/Variable_41*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_41/readIdentitytraining/Adam/Variable_41*
T0*,
_class"
 loc:@training/Adam/Variable_41*
_output_shapes
:
z
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
T0*&
_output_shapes
:@
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
:@
u
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*&
_output_shapes
:@*
T0
}
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0*&
_output_shapes
:@
Z
training/Adam/sub_3/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/SquareSquareFtraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
v
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*&
_output_shapes
:@
u
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*&
_output_shapes
:@
s
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*&
_output_shapes
:@
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
:@

training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
T0*&
_output_shapes
:@
l
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*&
_output_shapes
:@
Z
training/Adam/add_3/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
x
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*&
_output_shapes
:@*
T0
}
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0*&
_output_shapes
:@
z
training/Adam/sub_4Subconv2d_1/kernel/readtraining/Adam/truediv_1*
T0*&
_output_shapes
:@
а
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*)
_class
loc:@training/Adam/Variable*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
и
training/Adam/Assign_1Assigntraining/Adam/Variable_14training/Adam/add_2*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14
Ф
training/Adam/Assign_2Assignconv2d_1/kerneltraining/Adam/sub_4*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
_output_shapes
:@*
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
:@
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
:@
q
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_15/read*
T0*
_output_shapes
:@
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
training/Adam/Square_1Square9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
_output_shapes
:@*
T0
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
:@
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
:@
Z
training/Adam/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_5Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
_output_shapes
:@*
T0

training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
T0*
_output_shapes
:@
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
_output_shapes
:@*
T0
Z
training/Adam/add_6/yConst*
dtype0*
_output_shapes
: *
valueB
 *Пж3
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
_output_shapes
:@*
T0
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes
:@
l
training/Adam/sub_7Subconv2d_1/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes
:@
Ъ
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:@
Ь
training/Adam/Assign_4Assigntraining/Adam/Variable_15training/Adam/add_5*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:@
Д
training/Adam/Assign_5Assignconv2d_1/biastraining/Adam/sub_7*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias
}
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0*&
_output_shapes
:@@
Z
training/Adam/sub_8/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
T0*
_output_shapes
: 
Љ
training/Adam/mul_12Multraining/Adam/sub_8Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@@
w
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*&
_output_shapes
:@@
~
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_16/read*
T0*&
_output_shapes
:@@
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
training/Adam/Square_2SquareFtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
y
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*&
_output_shapes
:@@*
T0
w
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*&
_output_shapes
:@@
t
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*&
_output_shapes
:@@*
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
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*&
_output_shapes
:@@*
T0

training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*&
_output_shapes
:@@*
T0
l
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*&
_output_shapes
:@@*
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
:@@
~
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*&
_output_shapes
:@@*
T0
{
training/Adam/sub_10Subconv2d_2/kernel/readtraining/Adam/truediv_3*&
_output_shapes
:@@*
T0
ж
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*&
_output_shapes
:@@
и
training/Adam/Assign_7Assigntraining/Adam/Variable_16training/Adam/add_8*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*&
_output_shapes
:@@
Х
training/Adam/Assign_8Assignconv2d_2/kerneltraining/Adam/sub_10*&
_output_shapes
:@@*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(
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
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
_output_shapes
:@*
T0
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_17/read*
_output_shapes
:@*
T0
[
training/Adam/sub_12/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
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
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
:@
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
:@
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
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes
:@
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes
:@
m
training/Adam/sub_13Subconv2d_2/bias/readtraining/Adam/truediv_4*
_output_shapes
:@*
T0
Ы
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3
Ю
training/Adam/Assign_10Assigntraining/Adam/Variable_17training/Adam/add_11*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17
Ж
training/Adam/Assign_11Assignconv2d_2/biastraining/Adam/sub_13*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@
~
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0*'
_output_shapes
:@
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
Ћ
training/Adam/mul_22Multraining/Adam/sub_14Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
y
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*'
_output_shapes
:@

training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_18/read*
T0*'
_output_shapes
:@
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

training/Adam/Square_4SquareFtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
{
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*'
_output_shapes
:@*
T0
y
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*'
_output_shapes
:@*
T0
v
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*'
_output_shapes
:@*
T0
[
training/Adam/Const_10Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_11Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*
T0*'
_output_shapes
:@

training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*'
_output_shapes
:@*
T0
m
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*'
_output_shapes
:@
[
training/Adam/add_15/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
{
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*'
_output_shapes
:@*
T0

training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*'
_output_shapes
:@*
T0
|
training/Adam/sub_16Subconv2d_3/kernel/readtraining/Adam/truediv_5*
T0*'
_output_shapes
:@
й
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*'
_output_shapes
:@
л
training/Adam/Assign_13Assigntraining/Adam/Variable_18training/Adam/add_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*'
_output_shapes
:@
Ч
training/Adam/Assign_14Assignconv2d_3/kerneltraining/Adam/sub_16*'
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(
r
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
_output_shapes	
:*
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

training/Adam/mul_27Multraining/Adam/sub_179training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
m
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
_output_shapes	
:*
T0
s
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_19/read*
T0*
_output_shapes	
:
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

training/Adam/Square_5Square9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes	
:
m
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
_output_shapes	
:*
T0
j
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes	
:
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

%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
_output_shapes	
:*
T0

training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
_output_shapes	
:*
T0
a
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes	
:
[
training/Adam/add_18/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
T0*
_output_shapes	
:
t
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
_output_shapes	
:*
T0
n
training/Adam/sub_19Subconv2d_3/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes	
:
Э
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes	
:
Я
training/Adam/Assign_16Assigntraining/Adam/Variable_19training/Adam/add_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes	
:
З
training/Adam/Assign_17Assignconv2d_3/biastraining/Adam/sub_19*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(

training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*(
_output_shapes
:*
T0
[
training/Adam/sub_20/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
_output_shapes
: *
T0
Ќ
training/Adam/mul_32Multraining/Adam/sub_20Ftraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
z
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*(
_output_shapes
:

training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_20/read*
T0*(
_output_shapes
:
[
training/Adam/sub_21/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_6SquareFtraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
|
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*(
_output_shapes
:
z
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*(
_output_shapes
:
w
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*(
_output_shapes
:
[
training/Adam/Const_14Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_15Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_15*
T0*(
_output_shapes
:

training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*(
_output_shapes
:*
T0
n
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0*(
_output_shapes
:
[
training/Adam/add_21/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
|
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*(
_output_shapes
:*
T0

training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*(
_output_shapes
:*
T0
}
training/Adam/sub_22Subconv2d_4/kernel/readtraining/Adam/truediv_7*(
_output_shapes
:*
T0
к
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
м
training/Adam/Assign_19Assigntraining/Adam/Variable_20training/Adam/add_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(*(
_output_shapes
:
Ш
training/Adam/Assign_20Assignconv2d_4/kerneltraining/Adam/sub_22*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*(
_output_shapes
:
r
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
_output_shapes	
:*
T0
[
training/Adam/sub_23/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_37Multraining/Adam/sub_239training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
m
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
_output_shapes	
:*
T0
s
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_21/read*
_output_shapes	
:*
T0
[
training/Adam/sub_24/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_7Square9training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes	
:
m
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes	
:
j
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
T0*
_output_shapes	
:
[
training/Adam/Const_16Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_17Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_17*
T0*
_output_shapes	
:

training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_16*
_output_shapes	
:*
T0
a
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes	
:
[
training/Adam/add_24/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
_output_shapes	
:*
T0
t
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
T0*
_output_shapes	
:
n
training/Adam/sub_25Subconv2d_4/bias/readtraining/Adam/truediv_8*
T0*
_output_shapes	
:
Э
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes	
:
Я
training/Adam/Assign_22Assigntraining/Adam/Variable_21training/Adam/add_23*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(
З
training/Adam/Assign_23Assignconv2d_4/biastraining/Adam/sub_25*
use_locking(*
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(*
_output_shapes	
:
~
training/Adam/mul_41MulAdam/beta_1/readtraining/Adam/Variable_8/read*
T0*'
_output_shapes
:@
[
training/Adam/sub_26/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_26Subtraining/Adam/sub_26/xAdam/beta_1/read*
_output_shapes
: *
T0
Ћ
training/Adam/mul_42Multraining/Adam/sub_26Ftraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*'
_output_shapes
:@*
T0
y
training/Adam/add_25Addtraining/Adam/mul_41training/Adam/mul_42*'
_output_shapes
:@*
T0

training/Adam/mul_43MulAdam/beta_2/readtraining/Adam/Variable_22/read*'
_output_shapes
:@*
T0
[
training/Adam/sub_27/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_27Subtraining/Adam/sub_27/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_8SquareFtraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*'
_output_shapes
:@*
T0
{
training/Adam/mul_44Multraining/Adam/sub_27training/Adam/Square_8*
T0*'
_output_shapes
:@
y
training/Adam/add_26Addtraining/Adam/mul_43training/Adam/mul_44*
T0*'
_output_shapes
:@
v
training/Adam/mul_45Multraining/Adam/multraining/Adam/add_25*
T0*'
_output_shapes
:@
[
training/Adam/Const_18Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_19Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_9/MinimumMinimumtraining/Adam/add_26training/Adam/Const_19*
T0*'
_output_shapes
:@

training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_18*
T0*'
_output_shapes
:@
m
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
T0*'
_output_shapes
:@
[
training/Adam/add_27/yConst*
dtype0*
_output_shapes
: *
valueB
 *Пж3
{
training/Adam/add_27Addtraining/Adam/Sqrt_9training/Adam/add_27/y*
T0*'
_output_shapes
:@

training/Adam/truediv_9RealDivtraining/Adam/mul_45training/Adam/add_27*'
_output_shapes
:@*
T0
|
training/Adam/sub_28Subconv2d_5/kernel/readtraining/Adam/truediv_9*
T0*'
_output_shapes
:@
й
training/Adam/Assign_24Assigntraining/Adam/Variable_8training/Adam/add_25*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*'
_output_shapes
:@
л
training/Adam/Assign_25Assigntraining/Adam/Variable_22training/Adam/add_26*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*'
_output_shapes
:@
Ч
training/Adam/Assign_26Assignconv2d_5/kerneltraining/Adam/sub_28*
T0*"
_class
loc:@conv2d_5/kernel*
validate_shape(*'
_output_shapes
:@*
use_locking(
q
training/Adam/mul_46MulAdam/beta_1/readtraining/Adam/Variable_9/read*
_output_shapes
:@*
T0
[
training/Adam/sub_29/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_29Subtraining/Adam/sub_29/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_47Multraining/Adam/sub_299training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
l
training/Adam/add_28Addtraining/Adam/mul_46training/Adam/mul_47*
T0*
_output_shapes
:@
r
training/Adam/mul_48MulAdam/beta_2/readtraining/Adam/Variable_23/read*
_output_shapes
:@*
T0
[
training/Adam/sub_30/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_30Subtraining/Adam/sub_30/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_9Square9training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
n
training/Adam/mul_49Multraining/Adam/sub_30training/Adam/Square_9*
T0*
_output_shapes
:@
l
training/Adam/add_29Addtraining/Adam/mul_48training/Adam/mul_49*
T0*
_output_shapes
:@
i
training/Adam/mul_50Multraining/Adam/multraining/Adam/add_28*
T0*
_output_shapes
:@
[
training/Adam/Const_20Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_21Const*
dtype0*
_output_shapes
: *
valueB
 *  

&training/Adam/clip_by_value_10/MinimumMinimumtraining/Adam/add_29training/Adam/Const_21*
_output_shapes
:@*
T0

training/Adam/clip_by_value_10Maximum&training/Adam/clip_by_value_10/Minimumtraining/Adam/Const_20*
T0*
_output_shapes
:@
b
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
T0*
_output_shapes
:@
[
training/Adam/add_30/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_30Addtraining/Adam/Sqrt_10training/Adam/add_30/y*
T0*
_output_shapes
:@
t
training/Adam/truediv_10RealDivtraining/Adam/mul_50training/Adam/add_30*
T0*
_output_shapes
:@
n
training/Adam/sub_31Subconv2d_5/bias/readtraining/Adam/truediv_10*
T0*
_output_shapes
:@
Ь
training/Adam/Assign_27Assigntraining/Adam/Variable_9training/Adam/add_28*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:@
Ю
training/Adam/Assign_28Assigntraining/Adam/Variable_23training/Adam/add_29*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes
:@
Ж
training/Adam/Assign_29Assignconv2d_5/biastraining/Adam/sub_31*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias*
validate_shape(*
_output_shapes
:@
~
training/Adam/mul_51MulAdam/beta_1/readtraining/Adam/Variable_10/read*
T0*&
_output_shapes
:@@
[
training/Adam/sub_32/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_32Subtraining/Adam/sub_32/xAdam/beta_1/read*
T0*
_output_shapes
: 
Њ
training/Adam/mul_52Multraining/Adam/sub_32Ftraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@@
x
training/Adam/add_31Addtraining/Adam/mul_51training/Adam/mul_52*
T0*&
_output_shapes
:@@
~
training/Adam/mul_53MulAdam/beta_2/readtraining/Adam/Variable_24/read*&
_output_shapes
:@@*
T0
[
training/Adam/sub_33/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_33Subtraining/Adam/sub_33/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_10SquareFtraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@@
{
training/Adam/mul_54Multraining/Adam/sub_33training/Adam/Square_10*
T0*&
_output_shapes
:@@
x
training/Adam/add_32Addtraining/Adam/mul_53training/Adam/mul_54*&
_output_shapes
:@@*
T0
u
training/Adam/mul_55Multraining/Adam/multraining/Adam/add_31*&
_output_shapes
:@@*
T0
[
training/Adam/Const_22Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_23Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_11/MinimumMinimumtraining/Adam/add_32training/Adam/Const_23*
T0*&
_output_shapes
:@@

training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_22*&
_output_shapes
:@@*
T0
n
training/Adam/Sqrt_11Sqrttraining/Adam/clip_by_value_11*
T0*&
_output_shapes
:@@
[
training/Adam/add_33/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
{
training/Adam/add_33Addtraining/Adam/Sqrt_11training/Adam/add_33/y*&
_output_shapes
:@@*
T0

training/Adam/truediv_11RealDivtraining/Adam/mul_55training/Adam/add_33*&
_output_shapes
:@@*
T0
|
training/Adam/sub_34Subconv2d_6/kernel/readtraining/Adam/truediv_11*
T0*&
_output_shapes
:@@
к
training/Adam/Assign_30Assigntraining/Adam/Variable_10training/Adam/add_31*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*&
_output_shapes
:@@
к
training/Adam/Assign_31Assigntraining/Adam/Variable_24training/Adam/add_32*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(*&
_output_shapes
:@@
Ц
training/Adam/Assign_32Assignconv2d_6/kerneltraining/Adam/sub_34*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*&
_output_shapes
:@@*
use_locking(
r
training/Adam/mul_56MulAdam/beta_1/readtraining/Adam/Variable_11/read*
_output_shapes
:@*
T0
[
training/Adam/sub_35/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_35Subtraining/Adam/sub_35/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_57Multraining/Adam/sub_359training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
l
training/Adam/add_34Addtraining/Adam/mul_56training/Adam/mul_57*
_output_shapes
:@*
T0
r
training/Adam/mul_58MulAdam/beta_2/readtraining/Adam/Variable_25/read*
T0*
_output_shapes
:@
[
training/Adam/sub_36/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_36Subtraining/Adam/sub_36/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_11Square9training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
o
training/Adam/mul_59Multraining/Adam/sub_36training/Adam/Square_11*
_output_shapes
:@*
T0
l
training/Adam/add_35Addtraining/Adam/mul_58training/Adam/mul_59*
T0*
_output_shapes
:@
i
training/Adam/mul_60Multraining/Adam/multraining/Adam/add_34*
_output_shapes
:@*
T0
[
training/Adam/Const_24Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_25Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_12/MinimumMinimumtraining/Adam/add_35training/Adam/Const_25*
_output_shapes
:@*
T0

training/Adam/clip_by_value_12Maximum&training/Adam/clip_by_value_12/Minimumtraining/Adam/Const_24*
T0*
_output_shapes
:@
b
training/Adam/Sqrt_12Sqrttraining/Adam/clip_by_value_12*
T0*
_output_shapes
:@
[
training/Adam/add_36/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_36Addtraining/Adam/Sqrt_12training/Adam/add_36/y*
_output_shapes
:@*
T0
t
training/Adam/truediv_12RealDivtraining/Adam/mul_60training/Adam/add_36*
T0*
_output_shapes
:@
n
training/Adam/sub_37Subconv2d_6/bias/readtraining/Adam/truediv_12*
_output_shapes
:@*
T0
Ю
training/Adam/Assign_33Assigntraining/Adam/Variable_11training/Adam/add_34*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(
Ю
training/Adam/Assign_34Assigntraining/Adam/Variable_25training/Adam/add_35*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25
Ж
training/Adam/Assign_35Assignconv2d_6/biastraining/Adam/sub_37*
use_locking(*
T0* 
_class
loc:@conv2d_6/bias*
validate_shape(*
_output_shapes
:@
~
training/Adam/mul_61MulAdam/beta_1/readtraining/Adam/Variable_12/read*
T0*&
_output_shapes
:@
[
training/Adam/sub_38/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_38Subtraining/Adam/sub_38/xAdam/beta_1/read*
_output_shapes
: *
T0
Њ
training/Adam/mul_62Multraining/Adam/sub_38Ftraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
x
training/Adam/add_37Addtraining/Adam/mul_61training/Adam/mul_62*&
_output_shapes
:@*
T0
~
training/Adam/mul_63MulAdam/beta_2/readtraining/Adam/Variable_26/read*
T0*&
_output_shapes
:@
[
training/Adam/sub_39/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_39Subtraining/Adam/sub_39/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_12SquareFtraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@*
T0
{
training/Adam/mul_64Multraining/Adam/sub_39training/Adam/Square_12*
T0*&
_output_shapes
:@
x
training/Adam/add_38Addtraining/Adam/mul_63training/Adam/mul_64*
T0*&
_output_shapes
:@
u
training/Adam/mul_65Multraining/Adam/multraining/Adam/add_37*
T0*&
_output_shapes
:@
[
training/Adam/Const_26Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_27Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_13/MinimumMinimumtraining/Adam/add_38training/Adam/Const_27*
T0*&
_output_shapes
:@

training/Adam/clip_by_value_13Maximum&training/Adam/clip_by_value_13/Minimumtraining/Adam/Const_26*&
_output_shapes
:@*
T0
n
training/Adam/Sqrt_13Sqrttraining/Adam/clip_by_value_13*&
_output_shapes
:@*
T0
[
training/Adam/add_39/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
{
training/Adam/add_39Addtraining/Adam/Sqrt_13training/Adam/add_39/y*
T0*&
_output_shapes
:@

training/Adam/truediv_13RealDivtraining/Adam/mul_65training/Adam/add_39*&
_output_shapes
:@*
T0
|
training/Adam/sub_40Subconv2d_7/kernel/readtraining/Adam/truediv_13*
T0*&
_output_shapes
:@
к
training/Adam/Assign_36Assigntraining/Adam/Variable_12training/Adam/add_37*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
к
training/Adam/Assign_37Assigntraining/Adam/Variable_26training/Adam/add_38*,
_class"
 loc:@training/Adam/Variable_26*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
Ц
training/Adam/Assign_38Assignconv2d_7/kerneltraining/Adam/sub_40*
use_locking(*
T0*"
_class
loc:@conv2d_7/kernel*
validate_shape(*&
_output_shapes
:@
r
training/Adam/mul_66MulAdam/beta_1/readtraining/Adam/Variable_13/read*
_output_shapes
:*
T0
[
training/Adam/sub_41/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_41Subtraining/Adam/sub_41/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_67Multraining/Adam/sub_419training/Adam/gradients/conv2d_7/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
l
training/Adam/add_40Addtraining/Adam/mul_66training/Adam/mul_67*
T0*
_output_shapes
:
r
training/Adam/mul_68MulAdam/beta_2/readtraining/Adam/Variable_27/read*
T0*
_output_shapes
:
[
training/Adam/sub_42/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_42Subtraining/Adam/sub_42/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_13Square9training/Adam/gradients/conv2d_7/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
o
training/Adam/mul_69Multraining/Adam/sub_42training/Adam/Square_13*
_output_shapes
:*
T0
l
training/Adam/add_41Addtraining/Adam/mul_68training/Adam/mul_69*
T0*
_output_shapes
:
i
training/Adam/mul_70Multraining/Adam/multraining/Adam/add_40*
T0*
_output_shapes
:
[
training/Adam/Const_28Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_29Const*
dtype0*
_output_shapes
: *
valueB
 *  

&training/Adam/clip_by_value_14/MinimumMinimumtraining/Adam/add_41training/Adam/Const_29*
_output_shapes
:*
T0

training/Adam/clip_by_value_14Maximum&training/Adam/clip_by_value_14/Minimumtraining/Adam/Const_28*
T0*
_output_shapes
:
b
training/Adam/Sqrt_14Sqrttraining/Adam/clip_by_value_14*
T0*
_output_shapes
:
[
training/Adam/add_42/yConst*
dtype0*
_output_shapes
: *
valueB
 *Пж3
o
training/Adam/add_42Addtraining/Adam/Sqrt_14training/Adam/add_42/y*
_output_shapes
:*
T0
t
training/Adam/truediv_14RealDivtraining/Adam/mul_70training/Adam/add_42*
T0*
_output_shapes
:
n
training/Adam/sub_43Subconv2d_7/bias/readtraining/Adam/truediv_14*
_output_shapes
:*
T0
Ю
training/Adam/Assign_39Assigntraining/Adam/Variable_13training/Adam/add_40*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:
Ю
training/Adam/Assign_40Assigntraining/Adam/Variable_27training/Adam/add_41*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_27*
validate_shape(*
_output_shapes
:
Ж
training/Adam/Assign_41Assignconv2d_7/biastraining/Adam/sub_43*
use_locking(*
T0* 
_class
loc:@conv2d_7/bias*
validate_shape(*
_output_shapes
:
ј
training/group_depsNoOp	^loss/mul^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_24^training/Adam/Assign_25^training/Adam/Assign_26^training/Adam/Assign_27^training/Adam/Assign_28^training/Adam/Assign_29^training/Adam/Assign_3^training/Adam/Assign_30^training/Adam/Assign_31^training/Adam/Assign_32^training/Adam/Assign_33^training/Adam/Assign_34^training/Adam/Assign_35^training/Adam/Assign_36^training/Adam/Assign_37^training/Adam/Assign_38^training/Adam/Assign_39^training/Adam/Assign_4^training/Adam/Assign_40^training/Adam/Assign_41^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9


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
IsVariableInitialized_2IsVariableInitializedconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_3IsVariableInitializedconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializedconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializedconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_6IsVariableInitializedconv2d_4/kernel*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_4/kernel

IsVariableInitialized_7IsVariableInitializedconv2d_4/bias* 
_class
loc:@conv2d_4/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializedconv2d_5/kernel*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_5/kernel

IsVariableInitialized_9IsVariableInitializedconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitializedconv2d_6/kernel*
_output_shapes
: *"
_class
loc:@conv2d_6/kernel*
dtype0

IsVariableInitialized_11IsVariableInitializedconv2d_6/bias* 
_class
loc:@conv2d_6/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_12IsVariableInitializedconv2d_7/kernel*"
_class
loc:@conv2d_7/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_13IsVariableInitializedconv2d_7/bias*
_output_shapes
: * 
_class
loc:@conv2d_7/bias*
dtype0

IsVariableInitialized_14IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
{
IsVariableInitialized_15IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 

IsVariableInitialized_16IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_17IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_18IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 

IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
dtype0*
_output_shapes
: 

IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_2*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_2*
dtype0

IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_3*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_3

IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_4*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_4*
dtype0

IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
: 

IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes
: 

IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0*
_output_shapes
: 

IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes
: 

IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_9*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_9*
dtype0

IsVariableInitialized_29IsVariableInitializedtraining/Adam/Variable_10*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_10*
dtype0

IsVariableInitialized_30IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
: 

IsVariableInitialized_31IsVariableInitializedtraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
: 

IsVariableInitialized_32IsVariableInitializedtraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
dtype0*
_output_shapes
: 

IsVariableInitialized_33IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes
: 

IsVariableInitialized_34IsVariableInitializedtraining/Adam/Variable_15*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_15*
dtype0

IsVariableInitialized_35IsVariableInitializedtraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0*
_output_shapes
: 

IsVariableInitialized_36IsVariableInitializedtraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
dtype0*
_output_shapes
: 

IsVariableInitialized_37IsVariableInitializedtraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
dtype0*
_output_shapes
: 

IsVariableInitialized_38IsVariableInitializedtraining/Adam/Variable_19*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_19

IsVariableInitialized_39IsVariableInitializedtraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0*
_output_shapes
: 

IsVariableInitialized_40IsVariableInitializedtraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
: 

IsVariableInitialized_41IsVariableInitializedtraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
dtype0*
_output_shapes
: 

IsVariableInitialized_42IsVariableInitializedtraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
dtype0*
_output_shapes
: 

IsVariableInitialized_43IsVariableInitializedtraining/Adam/Variable_24*,
_class"
 loc:@training/Adam/Variable_24*
dtype0*
_output_shapes
: 

IsVariableInitialized_44IsVariableInitializedtraining/Adam/Variable_25*,
_class"
 loc:@training/Adam/Variable_25*
dtype0*
_output_shapes
: 

IsVariableInitialized_45IsVariableInitializedtraining/Adam/Variable_26*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_26

IsVariableInitialized_46IsVariableInitializedtraining/Adam/Variable_27*,
_class"
 loc:@training/Adam/Variable_27*
dtype0*
_output_shapes
: 

IsVariableInitialized_47IsVariableInitializedtraining/Adam/Variable_28*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_28*
dtype0

IsVariableInitialized_48IsVariableInitializedtraining/Adam/Variable_29*,
_class"
 loc:@training/Adam/Variable_29*
dtype0*
_output_shapes
: 

IsVariableInitialized_49IsVariableInitializedtraining/Adam/Variable_30*,
_class"
 loc:@training/Adam/Variable_30*
dtype0*
_output_shapes
: 

IsVariableInitialized_50IsVariableInitializedtraining/Adam/Variable_31*,
_class"
 loc:@training/Adam/Variable_31*
dtype0*
_output_shapes
: 

IsVariableInitialized_51IsVariableInitializedtraining/Adam/Variable_32*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_32*
dtype0

IsVariableInitialized_52IsVariableInitializedtraining/Adam/Variable_33*,
_class"
 loc:@training/Adam/Variable_33*
dtype0*
_output_shapes
: 

IsVariableInitialized_53IsVariableInitializedtraining/Adam/Variable_34*,
_class"
 loc:@training/Adam/Variable_34*
dtype0*
_output_shapes
: 

IsVariableInitialized_54IsVariableInitializedtraining/Adam/Variable_35*,
_class"
 loc:@training/Adam/Variable_35*
dtype0*
_output_shapes
: 

IsVariableInitialized_55IsVariableInitializedtraining/Adam/Variable_36*,
_class"
 loc:@training/Adam/Variable_36*
dtype0*
_output_shapes
: 

IsVariableInitialized_56IsVariableInitializedtraining/Adam/Variable_37*,
_class"
 loc:@training/Adam/Variable_37*
dtype0*
_output_shapes
: 

IsVariableInitialized_57IsVariableInitializedtraining/Adam/Variable_38*,
_class"
 loc:@training/Adam/Variable_38*
dtype0*
_output_shapes
: 

IsVariableInitialized_58IsVariableInitializedtraining/Adam/Variable_39*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_39

IsVariableInitialized_59IsVariableInitializedtraining/Adam/Variable_40*,
_class"
 loc:@training/Adam/Variable_40*
dtype0*
_output_shapes
: 

IsVariableInitialized_60IsVariableInitializedtraining/Adam/Variable_41*,
_class"
 loc:@training/Adam/Variable_41*
dtype0*
_output_shapes
: 
і
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^conv2d_3/bias/Assign^conv2d_3/kernel/Assign^conv2d_4/bias/Assign^conv2d_4/kernel/Assign^conv2d_5/bias/Assign^conv2d_5/kernel/Assign^conv2d_6/bias/Assign^conv2d_6/kernel/Assign^conv2d_7/bias/Assign^conv2d_7/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign!^training/Adam/Variable_24/Assign!^training/Adam/Variable_25/Assign!^training/Adam/Variable_26/Assign!^training/Adam/Variable_27/Assign!^training/Adam/Variable_28/Assign!^training/Adam/Variable_29/Assign ^training/Adam/Variable_3/Assign!^training/Adam/Variable_30/Assign!^training/Adam/Variable_31/Assign!^training/Adam/Variable_32/Assign!^training/Adam/Variable_33/Assign!^training/Adam/Variable_34/Assign!^training/Adam/Variable_35/Assign!^training/Adam/Variable_36/Assign!^training/Adam/Variable_37/Assign!^training/Adam/Variable_38/Assign!^training/Adam/Variable_39/Assign ^training/Adam/Variable_4/Assign!^training/Adam/Variable_40/Assign!^training/Adam/Variable_41/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign"у]|zvn     Ді)	фDыЙ зAJщм
й.З.
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

DepthToSpace

input"T
output"T"	
Ttype"

block_sizeint(0":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
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
q
ResizeBilinear
images"T
size
resized_images"
Ttype:
2
	"
align_cornersbool( 
q
ResizeBilinearGrad	
grads
original_image"T
output"T"
Ttype:
2"
align_cornersbool( 
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

SpaceToDepth

input"T
output"T"	
Ttype"

block_sizeint(0":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
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
shared_namestring *1.13.12v1.13.1-0-g6612da8951ђЃ

conv2d_1_inputPlaceholder*&
shape:џџџџџџџџџИа*
dtype0*1
_output_shapes
:џџџџџџџџџИа
v
conv2d_1/random_uniform/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
`
conv2d_1/random_uniform/minConst*
valueB
 *8JЬН*
dtype0*
_output_shapes
: 
`
conv2d_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *8JЬ=
В
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
T0*
dtype0*
seed2ШЄЧ*&
_output_shapes
:@*
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
:@

conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*
T0*&
_output_shapes
:@

conv2d_1/kernel
VariableV2*
shape:@*
shared_name *
dtype0*
	container *&
_output_shapes
:@
Ш
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@

conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
[
conv2d_1/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_1/bias
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:@*
shape:@
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
dtype0*
_output_shapes
:*
valueB"      
ю
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа@*
	dilations
*
T0

conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа@

leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_1/BiasAdd*
alpha%>*1
_output_shapes
:џџџџџџџџџИа@*
T0
v
conv2d_2/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
`
conv2d_2/random_uniform/minConst*
valueB
 *:ЭН*
dtype0*
_output_shapes
: 
`
conv2d_2/random_uniform/maxConst*
valueB
 *:Э=*
dtype0*
_output_shapes
: 
В
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
seedБџх)*
T0*
dtype0*
seed2ИЕ*&
_output_shapes
:@@
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
T0*
_output_shapes
: 

conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*
T0*&
_output_shapes
:@@

conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*&
_output_shapes
:@@*
T0

conv2d_2/kernel
VariableV2*
shared_name *
dtype0*
	container *&
_output_shapes
:@@*
shape:@@
Ш
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0

conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
[
conv2d_2/ConstConst*
_output_shapes
:@*
valueB@*    *
dtype0
y
conv2d_2/bias
VariableV2*
shape:@*
shared_name *
dtype0*
	container *
_output_shapes
:@
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
dtype0*
_output_shapes
:*
valueB"      
ї
conv2d_2/convolutionConv2Dleaky_re_lu_1/LeakyReluconv2d_2/kernel/read*1
_output_shapes
:џџџџџџџџџИа@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа@

leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_2/BiasAdd*
T0*
alpha%>*1
_output_shapes
:џџџџџџџџџИа@
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
dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*1
_output_shapes
:џџџџџџџџџИа@*
T0
й
dropout_1/cond/mul/SwitchSwitchleaky_re_lu_2/LeakyReludropout_1/cond/pred_id*N
_output_shapes<
::џџџџџџџџџИа@:џџџџџџџџџИа@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu
z
dropout_1/cond/dropout/rateConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ>
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
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
seed2иЂ*1
_output_shapes
:џџџџџџџџџИа@*
seedБџх)*
T0*
dtype0
Ї
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ь
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*1
_output_shapes
:џџџџџџџџџИа@
О
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*1
_output_shapes
:џџџџџџџџџИа@
 
dropout_1/cond/dropout/addAdddropout_1/cond/dropout/sub%dropout_1/cond/dropout/random_uniform*1
_output_shapes
:џџџџџџџџџИа@*
T0
}
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*
T0*1
_output_shapes
:џџџџџџџџџИа@

dropout_1/cond/dropout/truedivRealDivdropout_1/cond/muldropout_1/cond/dropout/sub*
T0*1
_output_shapes
:џџџџџџџџџИа@

dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/truedivdropout_1/cond/dropout/Floor*
T0*1
_output_shapes
:џџџџџџџџџИа@
з
dropout_1/cond/Switch_1Switchleaky_re_lu_2/LeakyReludropout_1/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*N
_output_shapes<
::џџџџџџџџџИа@:џџџџџџџџџИа@

dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
N*3
_output_shapes!
:џџџџџџџџџИа@: *
T0
Ц
max_pooling2d_1/MaxPoolMaxPooldropout_1/cond/Merge*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ@*
T0
v
conv2d_3/random_uniform/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
`
conv2d_3/random_uniform/minConst*
valueB
 *я[qН*
dtype0*
_output_shapes
: 
`
conv2d_3/random_uniform/maxConst*
valueB
 *я[q=*
dtype0*
_output_shapes
: 
Г
%conv2d_3/random_uniform/RandomUniformRandomUniformconv2d_3/random_uniform/shape*
seedБџх)*
T0*
dtype0*
seed2єј*'
_output_shapes
:@
}
conv2d_3/random_uniform/subSubconv2d_3/random_uniform/maxconv2d_3/random_uniform/min*
T0*
_output_shapes
: 

conv2d_3/random_uniform/mulMul%conv2d_3/random_uniform/RandomUniformconv2d_3/random_uniform/sub*
T0*'
_output_shapes
:@

conv2d_3/random_uniformAddconv2d_3/random_uniform/mulconv2d_3/random_uniform/min*
T0*'
_output_shapes
:@

conv2d_3/kernel
VariableV2*
	container *'
_output_shapes
:@*
shape:@*
shared_name *
dtype0
Щ
conv2d_3/kernel/AssignAssignconv2d_3/kernelconv2d_3/random_uniform*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel

conv2d_3/kernel/readIdentityconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*'
_output_shapes
:@*
T0
]
conv2d_3/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
{
conv2d_3/bias
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
Ў
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/Const*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes	
:
u
conv2d_3/bias/readIdentityconv2d_3/bias*
_output_shapes	
:*
T0* 
_class
loc:@conv2d_3/bias
s
"conv2d_3/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ј
conv2d_3/convolutionConv2Dmax_pooling2d_1/MaxPoolconv2d_3/kernel/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:џџџџџџџџџЈ*
	dilations
*
T0

conv2d_3/BiasAddBiasAddconv2d_3/convolutionconv2d_3/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:џџџџџџџџџЈ

leaky_re_lu_3/LeakyRelu	LeakyReluconv2d_3/BiasAdd*
alpha%>*2
_output_shapes 
:џџџџџџџџџЈ*
T0
v
conv2d_4/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
`
conv2d_4/random_uniform/minConst*
valueB
 *ьQН*
dtype0*
_output_shapes
: 
`
conv2d_4/random_uniform/maxConst*
valueB
 *ьQ=*
dtype0*
_output_shapes
: 
Г
%conv2d_4/random_uniform/RandomUniformRandomUniformconv2d_4/random_uniform/shape*
seedБџх)*
T0*
dtype0*
seed2ЩЧ+*(
_output_shapes
:
}
conv2d_4/random_uniform/subSubconv2d_4/random_uniform/maxconv2d_4/random_uniform/min*
_output_shapes
: *
T0

conv2d_4/random_uniform/mulMul%conv2d_4/random_uniform/RandomUniformconv2d_4/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_4/random_uniformAddconv2d_4/random_uniform/mulconv2d_4/random_uniform/min*
T0*(
_output_shapes
:

conv2d_4/kernel
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
Ъ
conv2d_4/kernel/AssignAssignconv2d_4/kernelconv2d_4/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*(
_output_shapes
:

conv2d_4/kernel/readIdentityconv2d_4/kernel*
T0*"
_class
loc:@conv2d_4/kernel*(
_output_shapes
:
]
conv2d_4/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_4/bias
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ў
conv2d_4/bias/AssignAssignconv2d_4/biasconv2d_4/Const*
use_locking(*
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(*
_output_shapes	
:
u
conv2d_4/bias/readIdentityconv2d_4/bias*
_output_shapes	
:*
T0* 
_class
loc:@conv2d_4/bias
s
"conv2d_4/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
ј
conv2d_4/convolutionConv2Dleaky_re_lu_3/LeakyReluconv2d_4/kernel/read*2
_output_shapes 
:џџџџџџџџџЈ*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME

conv2d_4/BiasAddBiasAddconv2d_4/convolutionconv2d_4/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:џџџџџџџџџЈ

leaky_re_lu_4/LeakyRelu	LeakyReluconv2d_4/BiasAdd*
alpha%>*2
_output_shapes 
:џџџџџџџџџЈ*
T0

dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes
: : *
T0

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

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*2
_output_shapes 
:џџџџџџџџџЈ*
T0
л
dropout_2/cond/mul/SwitchSwitchleaky_re_lu_4/LeakyReludropout_2/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*P
_output_shapes>
<:џџџџџџџџџЈ:џџџџџџџџџЈ
z
dropout_2/cond/dropout/rateConst^dropout_2/cond/switch_t*
valueB
 *ЭЬЬ>*
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
seedБџх)*
T0*
dtype0*
seed2Бњ*2
_output_shapes 
:џџџџџџџџџЈ
Ї
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
Э
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*
T0*2
_output_shapes 
:џџџџџџџџџЈ
П
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*
T0*2
_output_shapes 
:џџџџџџџџџЈ
Ё
dropout_2/cond/dropout/addAdddropout_2/cond/dropout/sub%dropout_2/cond/dropout/random_uniform*
T0*2
_output_shapes 
:џџџџџџџџџЈ
~
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*
T0*2
_output_shapes 
:џџџџџџџџџЈ

dropout_2/cond/dropout/truedivRealDivdropout_2/cond/muldropout_2/cond/dropout/sub*
T0*2
_output_shapes 
:џџџџџџџџџЈ

dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/truedivdropout_2/cond/dropout/Floor*2
_output_shapes 
:џџџџџџџџџЈ*
T0
й
dropout_2/cond/Switch_1Switchleaky_re_lu_4/LeakyReludropout_2/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*P
_output_shapes>
<:џџџџџџџџџЈ:џџџџџџџџџЈ

dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
T0*
N*4
_output_shapes"
 :џџџџџџџџџЈ: 
Ч
max_pooling2d_2/MaxPoolMaxPooldropout_2/cond/Merge*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*2
_output_shapes 
:џџџџџџџџџ
l
up_sampling2d_1/ShapeShapemax_pooling2d_2/MaxPool*
_output_shapes
:*
T0*
out_type0
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
up_sampling2d_1/mulMulup_sampling2d_1/strided_sliceup_sampling2d_1/Const*
_output_shapes
:*
T0
О
%up_sampling2d_1/ResizeNearestNeighborResizeNearestNeighbormax_pooling2d_2/MaxPoolup_sampling2d_1/mul*
align_corners( *
T0*2
_output_shapes 
:џџџџџџџџџЈ
v
conv2d_5/random_uniform/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
`
conv2d_5/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *я[qН
`
conv2d_5/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *я[q=
Г
%conv2d_5/random_uniform/RandomUniformRandomUniformconv2d_5/random_uniform/shape*
dtype0*
seed2Џ*'
_output_shapes
:@*
seedБџх)*
T0
}
conv2d_5/random_uniform/subSubconv2d_5/random_uniform/maxconv2d_5/random_uniform/min*
_output_shapes
: *
T0

conv2d_5/random_uniform/mulMul%conv2d_5/random_uniform/RandomUniformconv2d_5/random_uniform/sub*
T0*'
_output_shapes
:@

conv2d_5/random_uniformAddconv2d_5/random_uniform/mulconv2d_5/random_uniform/min*
T0*'
_output_shapes
:@

conv2d_5/kernel
VariableV2*
dtype0*
	container *'
_output_shapes
:@*
shape:@*
shared_name 
Щ
conv2d_5/kernel/AssignAssignconv2d_5/kernelconv2d_5/random_uniform*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel

conv2d_5/kernel/readIdentityconv2d_5/kernel*
T0*"
_class
loc:@conv2d_5/kernel*'
_output_shapes
:@
[
conv2d_5/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_5/bias
VariableV2*
shape:@*
shared_name *
dtype0*
	container *
_output_shapes
:@
­
conv2d_5/bias/AssignAssignconv2d_5/biasconv2d_5/Const*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias*
validate_shape(*
_output_shapes
:@
t
conv2d_5/bias/readIdentityconv2d_5/bias*
_output_shapes
:@*
T0* 
_class
loc:@conv2d_5/bias
s
"conv2d_5/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

conv2d_5/convolutionConv2D%up_sampling2d_1/ResizeNearestNeighborconv2d_5/kernel/read*
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
conv2d_5/BiasAddBiasAddconv2d_5/convolutionconv2d_5/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџЈ@

leaky_re_lu_5/LeakyRelu	LeakyReluconv2d_5/BiasAdd*
T0*
alpha%>*1
_output_shapes
:џџџџџџџџџЈ@
v
conv2d_6/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
`
conv2d_6/random_uniform/minConst*
valueB
 *:ЭН*
dtype0*
_output_shapes
: 
`
conv2d_6/random_uniform/maxConst*
valueB
 *:Э=*
dtype0*
_output_shapes
: 
В
%conv2d_6/random_uniform/RandomUniformRandomUniformconv2d_6/random_uniform/shape*
T0*
dtype0*
seed2ЧІ*&
_output_shapes
:@@*
seedБџх)
}
conv2d_6/random_uniform/subSubconv2d_6/random_uniform/maxconv2d_6/random_uniform/min*
_output_shapes
: *
T0

conv2d_6/random_uniform/mulMul%conv2d_6/random_uniform/RandomUniformconv2d_6/random_uniform/sub*
T0*&
_output_shapes
:@@

conv2d_6/random_uniformAddconv2d_6/random_uniform/mulconv2d_6/random_uniform/min*&
_output_shapes
:@@*
T0

conv2d_6/kernel
VariableV2*
shape:@@*
shared_name *
dtype0*
	container *&
_output_shapes
:@@
Ш
conv2d_6/kernel/AssignAssignconv2d_6/kernelconv2d_6/random_uniform*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel

conv2d_6/kernel/readIdentityconv2d_6/kernel*
T0*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:@@
[
conv2d_6/ConstConst*
dtype0*
_output_shapes
:@*
valueB@*    
y
conv2d_6/bias
VariableV2*
shape:@*
shared_name *
dtype0*
	container *
_output_shapes
:@
­
conv2d_6/bias/AssignAssignconv2d_6/biasconv2d_6/Const* 
_class
loc:@conv2d_6/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
t
conv2d_6/bias/readIdentityconv2d_6/bias*
T0* 
_class
loc:@conv2d_6/bias*
_output_shapes
:@
s
"conv2d_6/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ї
conv2d_6/convolutionConv2Dleaky_re_lu_5/LeakyReluconv2d_6/kernel/read*
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
conv2d_6/BiasAddBiasAddconv2d_6/convolutionconv2d_6/bias/read*1
_output_shapes
:џџџџџџџџџЈ@*
T0*
data_formatNHWC

leaky_re_lu_6/LeakyRelu	LeakyReluconv2d_6/BiasAdd*
alpha%>*1
_output_shapes
:џџџџџџџџџЈ@*
T0

dropout_3/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_3/cond/switch_tIdentitydropout_3/cond/Switch:1*
_output_shapes
: *
T0

[
dropout_3/cond/switch_fIdentitydropout_3/cond/Switch*
T0
*
_output_shapes
: 
c
dropout_3/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
: *
T0

s
dropout_3/cond/mul/yConst^dropout_3/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 

dropout_3/cond/mulMuldropout_3/cond/mul/Switch:1dropout_3/cond/mul/y*1
_output_shapes
:џџџџџџџџџЈ@*
T0
й
dropout_3/cond/mul/SwitchSwitchleaky_re_lu_6/LeakyReludropout_3/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@
z
dropout_3/cond/dropout/rateConst^dropout_3/cond/switch_t*
valueB
 *ЭЬЬ>*
dtype0*
_output_shapes
: 
n
dropout_3/cond/dropout/ShapeShapedropout_3/cond/mul*
_output_shapes
:*
T0*
out_type0
{
dropout_3/cond/dropout/sub/xConst^dropout_3/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
}
dropout_3/cond/dropout/subSubdropout_3/cond/dropout/sub/xdropout_3/cond/dropout/rate*
T0*
_output_shapes
: 

)dropout_3/cond/dropout/random_uniform/minConst^dropout_3/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_3/cond/dropout/random_uniform/maxConst^dropout_3/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ъ
3dropout_3/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_3/cond/dropout/Shape*
seed2хо*1
_output_shapes
:џџџџџџџџџЈ@*
seedБџх)*
T0*
dtype0
Ї
)dropout_3/cond/dropout/random_uniform/subSub)dropout_3/cond/dropout/random_uniform/max)dropout_3/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ь
)dropout_3/cond/dropout/random_uniform/mulMul3dropout_3/cond/dropout/random_uniform/RandomUniform)dropout_3/cond/dropout/random_uniform/sub*
T0*1
_output_shapes
:џџџџџџџџџЈ@
О
%dropout_3/cond/dropout/random_uniformAdd)dropout_3/cond/dropout/random_uniform/mul)dropout_3/cond/dropout/random_uniform/min*
T0*1
_output_shapes
:џџџџџџџџџЈ@
 
dropout_3/cond/dropout/addAdddropout_3/cond/dropout/sub%dropout_3/cond/dropout/random_uniform*
T0*1
_output_shapes
:џџџџџџџџџЈ@
}
dropout_3/cond/dropout/FloorFloordropout_3/cond/dropout/add*1
_output_shapes
:џџџџџџџџџЈ@*
T0

dropout_3/cond/dropout/truedivRealDivdropout_3/cond/muldropout_3/cond/dropout/sub*
T0*1
_output_shapes
:џџџџџџџџџЈ@

dropout_3/cond/dropout/mulMuldropout_3/cond/dropout/truedivdropout_3/cond/dropout/Floor*
T0*1
_output_shapes
:џџџџџџџџџЈ@
з
dropout_3/cond/Switch_1Switchleaky_re_lu_6/LeakyReludropout_3/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@

dropout_3/cond/MergeMergedropout_3/cond/Switch_1dropout_3/cond/dropout/mul*
T0*
N*3
_output_shapes!
:џџџџџџџџџЈ@: 
v
conv2d_7/random_uniform/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
`
conv2d_7/random_uniform/minConst*
valueB
 *мО*
dtype0*
_output_shapes
: 
`
conv2d_7/random_uniform/maxConst*
valueB
 *м>*
dtype0*
_output_shapes
: 
В
%conv2d_7/random_uniform/RandomUniformRandomUniformconv2d_7/random_uniform/shape*
T0*
dtype0*
seed2юи*&
_output_shapes
:@*
seedБџх)
}
conv2d_7/random_uniform/subSubconv2d_7/random_uniform/maxconv2d_7/random_uniform/min*
T0*
_output_shapes
: 

conv2d_7/random_uniform/mulMul%conv2d_7/random_uniform/RandomUniformconv2d_7/random_uniform/sub*
T0*&
_output_shapes
:@

conv2d_7/random_uniformAddconv2d_7/random_uniform/mulconv2d_7/random_uniform/min*
T0*&
_output_shapes
:@

conv2d_7/kernel
VariableV2*
dtype0*
	container *&
_output_shapes
:@*
shape:@*
shared_name 
Ш
conv2d_7/kernel/AssignAssignconv2d_7/kernelconv2d_7/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_7/kernel*
validate_shape(*&
_output_shapes
:@

conv2d_7/kernel/readIdentityconv2d_7/kernel*&
_output_shapes
:@*
T0*"
_class
loc:@conv2d_7/kernel
[
conv2d_7/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv2d_7/bias
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
­
conv2d_7/bias/AssignAssignconv2d_7/biasconv2d_7/Const* 
_class
loc:@conv2d_7/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
t
conv2d_7/bias/readIdentityconv2d_7/bias*
T0* 
_class
loc:@conv2d_7/bias*
_output_shapes
:
s
"conv2d_7/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
є
conv2d_7/convolutionConv2Ddropout_3/cond/Mergeconv2d_7/kernel/read*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ*
	dilations
*
T0*
strides
*
data_formatNHWC

conv2d_7/BiasAddBiasAddconv2d_7/convolutionconv2d_7/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџЈ

lambda_1/DepthToSpaceDepthToSpaceconv2d_7/BiasAdd*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа*

block_size*
T0
m
lambda_1/ResizeBilinear/sizeConst*
valueB"8  P  *
dtype0*
_output_shapes
:
Џ
lambda_1/ResizeBilinearResizeBilinearlambda_1/DepthToSpacelambda_1/ResizeBilinear/size*1
_output_shapes
:џџџџџџџџџИа*
align_corners( *
T0

lambda_1/PlaceholderPlaceholder*
dtype0*1
_output_shapes
:џџџџџџџџџЈ*&
shape:џџџџџџџџџЈ
Ђ
lambda_1/DepthToSpace_1DepthToSpacelambda_1/Placeholder*

block_size*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџИа
o
lambda_1/ResizeBilinear_1/sizeConst*
valueB"8  P  *
dtype0*
_output_shapes
:
Е
lambda_1/ResizeBilinear_1ResizeBilinearlambda_1/DepthToSpace_1lambda_1/ResizeBilinear_1/size*1
_output_shapes
:џџџџџџџџџИа*
align_corners( *
T0
_
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
dtype0	*
	container *
_output_shapes
: *
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
VariableV2*
	container *
_output_shapes
: *
shape: *
shared_name *
dtype0
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
Adam/lr/readIdentityAdam/lr*
T0*
_class
loc:@Adam/lr*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*
dtype0
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
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_1*
validate_shape(
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_class
loc:@Adam/beta_1*
_output_shapes
: 
^
Adam/beta_2/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wО?
o
Adam/beta_2
VariableV2*
	container *
_output_shapes
: *
shape: *
shared_name *
dtype0
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
Adam/decay/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
n

Adam/decay
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
shape: 
Њ
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
use_locking(*
T0*
_class
loc:@Adam/decay*
validate_shape(*
_output_shapes
: 
g
Adam/decay/readIdentity
Adam/decay*
_output_shapes
: *
T0*
_class
loc:@Adam/decay
И
lambda_1_targetPlaceholder*
dtype0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*?
shape6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
r
lambda_1_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ

loss/lambda_1_loss/subSublambda_1/ResizeBilinearlambda_1_target*1
_output_shapes
:џџџџџџџџџИа*
T0
q
loss/lambda_1_loss/AbsAbsloss/lambda_1_loss/sub*1
_output_shapes
:џџџџџџџџџИа*
T0
t
)loss/lambda_1_loss/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
З
loss/lambda_1_loss/MeanMeanloss/lambda_1_loss/Abs)loss/lambda_1_loss/Mean/reduction_indices*-
_output_shapes
:џџџџџџџџџИа*

Tidx0*
	keep_dims( *
T0
|
+loss/lambda_1_loss/Mean_1/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
В
loss/lambda_1_loss/Mean_1Meanloss/lambda_1_loss/Mean+loss/lambda_1_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( 

loss/lambda_1_loss/mulMulloss/lambda_1_loss/Mean_1lambda_1_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
b
loss/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/lambda_1_loss/NotEqualNotEquallambda_1_sample_weightsloss/lambda_1_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ

loss/lambda_1_loss/CastCastloss/lambda_1_loss/NotEqual*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:џџџџџџџџџ
b
loss/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/lambda_1_loss/Mean_2Meanloss/lambda_1_loss/Castloss/lambda_1_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 

loss/lambda_1_loss/truedivRealDivloss/lambda_1_loss/mulloss/lambda_1_loss/Mean_2*#
_output_shapes
:џџџџџџџџџ*
T0
d
loss/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/lambda_1_loss/Mean_3Meanloss/lambda_1_loss/truedivloss/lambda_1_loss/Const_1*
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
loss/mul/xloss/lambda_1_loss/Mean_3*
_output_shapes
: *
T0
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
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
_class
loc:@loss/mul*

index_type0*
_output_shapes
: *
T0
Ї
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/lambda_1_loss/Mean_3*
T0*
_class
loc:@loss/mul*
_output_shapes
: 

+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
_output_shapes
: *
T0*
_class
loc:@loss/mul
М
Dtraining/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Reshape/shapeConst*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
valueB:*
dtype0*
_output_shapes
:

>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Dtraining/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Reshape/shape*
_output_shapes
:*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
Tshape0
Ф
<training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/ShapeShapeloss/lambda_1_loss/truediv*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
out_type0*
_output_shapes
:
Џ
;training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/TileTile>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Reshape<training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Shape*

Tmultiples0*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Ц
>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Shape_1Shapeloss/lambda_1_loss/truediv*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
out_type0*
_output_shapes
:
Џ
>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Shape_2Const*
dtype0*
_output_shapes
: *,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
valueB 
Д
<training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/ConstConst*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
valueB: *
dtype0*
_output_shapes
:
­
;training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/ProdProd>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Shape_1<training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Const*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
_output_shapes
: *
	keep_dims( *

Tidx0
Ж
>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Const_1Const*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
valueB: *
dtype0*
_output_shapes
:
Б
=training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Prod_1Prod>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Shape_2>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3
А
@training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Maximum/yConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
value	B :

>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/MaximumMaximum=training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Prod_1@training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Maximum/y*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*
_output_shapes
: 

?training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/floordivFloorDiv;training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Prod>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Maximum*
_output_shapes
: *
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3
ђ
;training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/CastCast?training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/floordiv*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3

>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/truedivRealDiv;training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Tile;training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/Cast*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Т
=training/Adam/gradients/loss/lambda_1_loss/truediv_grad/ShapeShapeloss/lambda_1_loss/mul*
_output_shapes
:*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*
out_type0
Б
?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Shape_1Const*-
_class#
!loc:@loss/lambda_1_loss/truediv*
valueB *
dtype0*
_output_shapes
: 
в
Mtraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Shape?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Shape_1*-
_class#
!loc:@loss/lambda_1_loss/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/RealDivRealDiv>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/truedivloss/lambda_1_loss/Mean_2*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*#
_output_shapes
:џџџџџџџџџ
С
;training/Adam/gradients/loss/lambda_1_loss/truediv_grad/SumSum?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/RealDivMtraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0
Б
?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/ReshapeReshape;training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Sum=training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Shape*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*
Tshape0*#
_output_shapes
:џџџџџџџџџ
З
;training/Adam/gradients/loss/lambda_1_loss/truediv_grad/NegNegloss/lambda_1_loss/mul*-
_class#
!loc:@loss/lambda_1_loss/truediv*#
_output_shapes
:џџџџџџџџџ*
T0

Atraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/RealDiv_1RealDiv;training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Negloss/lambda_1_loss/Mean_2*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Atraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/RealDiv_2RealDivAtraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/RealDiv_1loss/lambda_1_loss/Mean_2*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*#
_output_shapes
:џџџџџџџџџ
Ђ
;training/Adam/gradients/loss/lambda_1_loss/truediv_grad/mulMul>training/Adam/gradients/loss/lambda_1_loss/Mean_3_grad/truedivAtraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/RealDiv_2*#
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv
С
=training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Sum_1Sum;training/Adam/gradients/loss/lambda_1_loss/truediv_grad/mulOtraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0
Њ
Atraining/Adam/gradients/loss/lambda_1_loss/truediv_grad/Reshape_1Reshape=training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Sum_1?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Shape_1*
T0*-
_class#
!loc:@loss/lambda_1_loss/truediv*
Tshape0*
_output_shapes
: 
Н
9training/Adam/gradients/loss/lambda_1_loss/mul_grad/ShapeShapeloss/lambda_1_loss/Mean_1*
T0*)
_class
loc:@loss/lambda_1_loss/mul*
out_type0*
_output_shapes
:
Н
;training/Adam/gradients/loss/lambda_1_loss/mul_grad/Shape_1Shapelambda_1_sample_weights*
_output_shapes
:*
T0*)
_class
loc:@loss/lambda_1_loss/mul*
out_type0
Т
Itraining/Adam/gradients/loss/lambda_1_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/loss/lambda_1_loss/mul_grad/Shape;training/Adam/gradients/loss/lambda_1_loss/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*)
_class
loc:@loss/lambda_1_loss/mul
ё
7training/Adam/gradients/loss/lambda_1_loss/mul_grad/MulMul?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Reshapelambda_1_sample_weights*
T0*)
_class
loc:@loss/lambda_1_loss/mul*#
_output_shapes
:џџџџџџџџџ
­
7training/Adam/gradients/loss/lambda_1_loss/mul_grad/SumSum7training/Adam/gradients/loss/lambda_1_loss/mul_grad/MulItraining/Adam/gradients/loss/lambda_1_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/lambda_1_loss/mul
Ё
;training/Adam/gradients/loss/lambda_1_loss/mul_grad/ReshapeReshape7training/Adam/gradients/loss/lambda_1_loss/mul_grad/Sum9training/Adam/gradients/loss/lambda_1_loss/mul_grad/Shape*
T0*)
_class
loc:@loss/lambda_1_loss/mul*
Tshape0*#
_output_shapes
:џџџџџџџџџ
ѕ
9training/Adam/gradients/loss/lambda_1_loss/mul_grad/Mul_1Mulloss/lambda_1_loss/Mean_1?training/Adam/gradients/loss/lambda_1_loss/truediv_grad/Reshape*
T0*)
_class
loc:@loss/lambda_1_loss/mul*#
_output_shapes
:џџџџџџџџџ
Г
9training/Adam/gradients/loss/lambda_1_loss/mul_grad/Sum_1Sum9training/Adam/gradients/loss/lambda_1_loss/mul_grad/Mul_1Ktraining/Adam/gradients/loss/lambda_1_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/lambda_1_loss/mul
Ї
=training/Adam/gradients/loss/lambda_1_loss/mul_grad/Reshape_1Reshape9training/Adam/gradients/loss/lambda_1_loss/mul_grad/Sum_1;training/Adam/gradients/loss/lambda_1_loss/mul_grad/Shape_1*
T0*)
_class
loc:@loss/lambda_1_loss/mul*
Tshape0*#
_output_shapes
:џџџџџџџџџ
С
<training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/ShapeShapeloss/lambda_1_loss/Mean*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
out_type0*
_output_shapes
:
Ћ
;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/SizeConst*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
ў
:training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/addAdd+loss/lambda_1_loss/Mean_1/reduction_indices;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Size*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
:

:training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/modFloorMod:training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/add;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Size*
_output_shapes
:*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1
Ж
>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape_1Const*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
valueB:*
dtype0*
_output_shapes
:
В
Btraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/range/startConst*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
value	B : *
dtype0*
_output_shapes
: 
В
Btraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/range/deltaConst*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
р
<training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/rangeRangeBtraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/range/start;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/SizeBtraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/range/delta*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
:*

Tidx0
Б
Atraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Fill/valueConst*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
Ћ
;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/FillFill>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape_1Atraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Fill/value*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*

index_type0*
_output_shapes
:
І
Dtraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/DynamicStitchDynamicStitch<training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/range:training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/mod<training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Fill*
N*
_output_shapes
:*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1
А
@training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Maximum/yConst*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
Є
>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/MaximumMaximumDtraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/DynamicStitch@training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Maximum/y*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
:

?training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/floordivFloorDiv<training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Maximum*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
:
а
>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/ReshapeReshape;training/Adam/gradients/loss/lambda_1_loss/mul_grad/ReshapeDtraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/DynamicStitch*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
Tshape0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0
Ь
;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/TileTile>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Reshape?training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/floordiv*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*

Tmultiples0*
T0
У
>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape_2Shapeloss/lambda_1_loss/Mean*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
out_type0*
_output_shapes
:*
T0
Х
>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape_3Shapeloss/lambda_1_loss/Mean_1*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
out_type0*
_output_shapes
:
Д
<training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/ConstConst*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
­
;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/ProdProd>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape_2<training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Const*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
: *
	keep_dims( *

Tidx0
Ж
>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Const_1Const*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
Б
=training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Prod_1Prod>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Shape_3>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Const_1*
	keep_dims( *

Tidx0*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
: 
В
Btraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Maximum_1/yConst*
_output_shapes
: *,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
value	B :*
dtype0

@training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Maximum_1Maximum=training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Prod_1Btraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Maximum_1/y*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
: *
T0

Atraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/floordiv_1FloorDiv;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Prod@training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Maximum_1*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
_output_shapes
: 
є
;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/CastCastAtraining/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/floordiv_1*,
_class"
 loc:@loss/lambda_1_loss/Mean_1*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
Љ
>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/truedivRealDiv;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Tile;training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/Cast*-
_output_shapes
:џџџџџџџџџИа*
T0*,
_class"
 loc:@loss/lambda_1_loss/Mean_1
М
:training/Adam/gradients/loss/lambda_1_loss/Mean_grad/ShapeShapeloss/lambda_1_loss/Abs**
_class 
loc:@loss/lambda_1_loss/Mean*
out_type0*
_output_shapes
:*
T0
Ї
9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/SizeConst**
_class 
loc:@loss/lambda_1_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
ђ
8training/Adam/gradients/loss/lambda_1_loss/Mean_grad/addAdd)loss/lambda_1_loss/Mean/reduction_indices9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Size*
_output_shapes
: *
T0**
_class 
loc:@loss/lambda_1_loss/Mean

8training/Adam/gradients/loss/lambda_1_loss/Mean_grad/modFloorMod8training/Adam/gradients/loss/lambda_1_loss/Mean_grad/add9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Size**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
: *
T0
Ћ
<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape_1Const**
_class 
loc:@loss/lambda_1_loss/Mean*
valueB *
dtype0*
_output_shapes
: 
Ў
@training/Adam/gradients/loss/lambda_1_loss/Mean_grad/range/startConst*
_output_shapes
: **
_class 
loc:@loss/lambda_1_loss/Mean*
value	B : *
dtype0
Ў
@training/Adam/gradients/loss/lambda_1_loss/Mean_grad/range/deltaConst*
_output_shapes
: **
_class 
loc:@loss/lambda_1_loss/Mean*
value	B :*
dtype0
ж
:training/Adam/gradients/loss/lambda_1_loss/Mean_grad/rangeRange@training/Adam/gradients/loss/lambda_1_loss/Mean_grad/range/start9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Size@training/Adam/gradients/loss/lambda_1_loss/Mean_grad/range/delta*

Tidx0**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
:
­
?training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Fill/valueConst**
_class 
loc:@loss/lambda_1_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/FillFill<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape_1?training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Fill/value*
T0**
_class 
loc:@loss/lambda_1_loss/Mean*

index_type0*
_output_shapes
: 

Btraining/Adam/gradients/loss/lambda_1_loss/Mean_grad/DynamicStitchDynamicStitch:training/Adam/gradients/loss/lambda_1_loss/Mean_grad/range8training/Adam/gradients/loss/lambda_1_loss/Mean_grad/mod:training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Fill*
_output_shapes
:*
T0**
_class 
loc:@loss/lambda_1_loss/Mean*
N
Ќ
>training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Maximum/yConst**
_class 
loc:@loss/lambda_1_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/MaximumMaximumBtraining/Adam/gradients/loss/lambda_1_loss/Mean_grad/DynamicStitch>training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Maximum/y*
T0**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
:

=training/Adam/gradients/loss/lambda_1_loss/Mean_grad/floordivFloorDiv:training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Maximum*
_output_shapes
:*
T0**
_class 
loc:@loss/lambda_1_loss/Mean
к
<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/ReshapeReshape>training/Adam/gradients/loss/lambda_1_loss/Mean_1_grad/truedivBtraining/Adam/gradients/loss/lambda_1_loss/Mean_grad/DynamicStitch*
T0**
_class 
loc:@loss/lambda_1_loss/Mean*
Tshape0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
б
9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/TileTile<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Reshape=training/Adam/gradients/loss/lambda_1_loss/Mean_grad/floordiv*

Tmultiples0*
T0**
_class 
loc:@loss/lambda_1_loss/Mean*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
О
<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape_2Shapeloss/lambda_1_loss/Abs**
_class 
loc:@loss/lambda_1_loss/Mean*
out_type0*
_output_shapes
:*
T0
П
<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape_3Shapeloss/lambda_1_loss/Mean*
_output_shapes
:*
T0**
_class 
loc:@loss/lambda_1_loss/Mean*
out_type0
А
:training/Adam/gradients/loss/lambda_1_loss/Mean_grad/ConstConst**
_class 
loc:@loss/lambda_1_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
Ѕ
9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/ProdProd<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape_2:training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0**
_class 
loc:@loss/lambda_1_loss/Mean
В
<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Const_1Const**
_class 
loc:@loss/lambda_1_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
Љ
;training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Prod_1Prod<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Shape_3<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Const_1*
T0**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
: *
	keep_dims( *

Tidx0
Ў
@training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Maximum_1/yConst**
_class 
loc:@loss/lambda_1_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

>training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Maximum_1Maximum;training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Prod_1@training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Maximum_1/y*
T0**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
: 

?training/Adam/gradients/loss/lambda_1_loss/Mean_grad/floordiv_1FloorDiv9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Prod>training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Maximum_1*
T0**
_class 
loc:@loss/lambda_1_loss/Mean*
_output_shapes
: 
ю
9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/CastCast?training/Adam/gradients/loss/lambda_1_loss/Mean_grad/floordiv_1*

SrcT0**
_class 
loc:@loss/lambda_1_loss/Mean*
Truncate( *

DstT0*
_output_shapes
: 
Ѕ
<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/truedivRealDiv9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Tile9training/Adam/gradients/loss/lambda_1_loss/Mean_grad/Cast*
T0**
_class 
loc:@loss/lambda_1_loss/Mean*1
_output_shapes
:џџџџџџџџџИа
П
8training/Adam/gradients/loss/lambda_1_loss/Abs_grad/SignSignloss/lambda_1_loss/sub*
T0*)
_class
loc:@loss/lambda_1_loss/Abs*1
_output_shapes
:џџџџџџџџџИа

7training/Adam/gradients/loss/lambda_1_loss/Abs_grad/mulMul<training/Adam/gradients/loss/lambda_1_loss/Mean_grad/truediv8training/Adam/gradients/loss/lambda_1_loss/Abs_grad/Sign*
T0*)
_class
loc:@loss/lambda_1_loss/Abs*1
_output_shapes
:џџџџџџџџџИа
Л
9training/Adam/gradients/loss/lambda_1_loss/sub_grad/ShapeShapelambda_1/ResizeBilinear*
_output_shapes
:*
T0*)
_class
loc:@loss/lambda_1_loss/sub*
out_type0
Е
;training/Adam/gradients/loss/lambda_1_loss/sub_grad/Shape_1Shapelambda_1_target*
T0*)
_class
loc:@loss/lambda_1_loss/sub*
out_type0*
_output_shapes
:
Т
Itraining/Adam/gradients/loss/lambda_1_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/loss/lambda_1_loss/sub_grad/Shape;training/Adam/gradients/loss/lambda_1_loss/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*)
_class
loc:@loss/lambda_1_loss/sub
­
7training/Adam/gradients/loss/lambda_1_loss/sub_grad/SumSum7training/Adam/gradients/loss/lambda_1_loss/Abs_grad/mulItraining/Adam/gradients/loss/lambda_1_loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/lambda_1_loss/sub*
_output_shapes
:
Џ
;training/Adam/gradients/loss/lambda_1_loss/sub_grad/ReshapeReshape7training/Adam/gradients/loss/lambda_1_loss/sub_grad/Sum9training/Adam/gradients/loss/lambda_1_loss/sub_grad/Shape*
T0*)
_class
loc:@loss/lambda_1_loss/sub*
Tshape0*1
_output_shapes
:џџџџџџџџџИа
Б
9training/Adam/gradients/loss/lambda_1_loss/sub_grad/Sum_1Sum7training/Adam/gradients/loss/lambda_1_loss/Abs_grad/mulKtraining/Adam/gradients/loss/lambda_1_loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/lambda_1_loss/sub
Ч
7training/Adam/gradients/loss/lambda_1_loss/sub_grad/NegNeg9training/Adam/gradients/loss/lambda_1_loss/sub_grad/Sum_1*
T0*)
_class
loc:@loss/lambda_1_loss/sub*
_output_shapes
:
Ь
=training/Adam/gradients/loss/lambda_1_loss/sub_grad/Reshape_1Reshape7training/Adam/gradients/loss/lambda_1_loss/sub_grad/Neg;training/Adam/gradients/loss/lambda_1_loss/sub_grad/Shape_1*
T0*)
_class
loc:@loss/lambda_1_loss/sub*
Tshape0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ў
Gtraining/Adam/gradients/lambda_1/ResizeBilinear_grad/ResizeBilinearGradResizeBilinearGrad;training/Adam/gradients/loss/lambda_1_loss/sub_grad/Reshapelambda_1/DepthToSpace*
T0**
_class 
loc:@lambda_1/ResizeBilinear*1
_output_shapes
:џџџџџџџџџИа*
align_corners( 
Ї
?training/Adam/gradients/lambda_1/DepthToSpace_grad/SpaceToDepthSpaceToDepthGtraining/Adam/gradients/lambda_1/ResizeBilinear_grad/ResizeBilinearGrad*1
_output_shapes
:џџџџџџџџџЈ*

block_size*
T0*(
_class
loc:@lambda_1/DepthToSpace*
data_formatNHWC
ъ
9training/Adam/gradients/conv2d_7/BiasAdd_grad/BiasAddGradBiasAddGrad?training/Adam/gradients/lambda_1/DepthToSpace_grad/SpaceToDepth*
data_formatNHWC*
_output_shapes
:*
T0*#
_class
loc:@conv2d_7/BiasAdd
л
8training/Adam/gradients/conv2d_7/convolution_grad/ShapeNShapeNdropout_3/cond/Mergeconv2d_7/kernel/read*
T0*'
_class
loc:@conv2d_7/convolution*
out_type0*
N* 
_output_shapes
::
Р
Etraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_7/convolution_grad/ShapeNconv2d_7/kernel/read?training/Adam/gradients/lambda_1/DepthToSpace_grad/SpaceToDepth*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ@*
	dilations
*
T0*'
_class
loc:@conv2d_7/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Й
Ftraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdropout_3/cond/Merge:training/Adam/gradients/conv2d_7/convolution_grad/ShapeN:1?training/Adam/gradients/lambda_1/DepthToSpace_grad/SpaceToDepth*'
_class
loc:@conv2d_7/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@*
	dilations
*
T0
І
;training/Adam/gradients/dropout_3/cond/Merge_grad/cond_gradSwitchEtraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropInputdropout_3/cond/pred_id*
T0*'
_class
loc:@conv2d_7/convolution*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@
о
training/Adam/gradients/SwitchSwitchleaky_re_lu_6/LeakyReludropout_3/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@
Ж
 training/Adam/gradients/IdentityIdentity training/Adam/gradients/Switch:1*1
_output_shapes
:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu
Ћ
training/Adam/gradients/Shape_1Shape training/Adam/gradients/Switch:1*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*
out_type0*
_output_shapes
:
З
#training/Adam/gradients/zeros/ConstConst!^training/Adam/gradients/Identity**
_class 
loc:@leaky_re_lu_6/LeakyRelu*
valueB
 *    *
dtype0*
_output_shapes
: 
х
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*

index_type0*1
_output_shapes
:џџџџџџџџџЈ@

>training/Adam/gradients/dropout_3/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_3/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџЈ@: 
Ъ
=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/ShapeShapedropout_3/cond/dropout/truediv*
_output_shapes
:*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
out_type0
Ъ
?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape_1Shapedropout_3/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
out_type0*
_output_shapes
:
в
Mtraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul

;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_3/cond/Merge_grad/cond_grad:1dropout_3/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџЈ@
Н
;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
_output_shapes
:
П
?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
Tshape0*1
_output_shapes
:џџџџџџџџџЈ@

=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Mul_1Muldropout_3/cond/dropout/truediv=training/Adam/gradients/dropout_3/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџЈ@
У
=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul
Х
Atraining/Adam/gradients/dropout_3/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Shape_1*1
_output_shapes
:џџџџџџџџџЈ@*
T0*-
_class#
!loc:@dropout_3/cond/dropout/mul*
Tshape0
Ц
Atraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/ShapeShapedropout_3/cond/mul*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*
out_type0*
_output_shapes
:
Й
Ctraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Shape_1Const*
_output_shapes
: *1
_class'
%#loc:@dropout_3/cond/dropout/truediv*
valueB *
dtype0
т
Qtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsAtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/ShapeCtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ctraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/RealDivRealDiv?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/Reshapedropout_3/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџЈ@*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv
б
?training/Adam/gradients/dropout_3/cond/dropout/truediv_grad/SumSumCtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/RealDivQtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*
_output_shapes
:
Я
Ctraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/ReshapeReshape?training/Adam/gradients/dropout_3/cond/dropout/truediv_grad/SumAtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Shape*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*
Tshape0*1
_output_shapes
:џџџџџџџџџЈ@
Щ
?training/Adam/gradients/dropout_3/cond/dropout/truediv_grad/NegNegdropout_3/cond/mul*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@

Etraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/RealDiv_1RealDiv?training/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Negdropout_3/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџЈ@*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv
Ђ
Etraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/RealDiv_2RealDivEtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/RealDiv_1dropout_3/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџЈ@*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv
Н
?training/Adam/gradients/dropout_3/cond/dropout/truediv_grad/mulMul?training/Adam/gradients/dropout_3/cond/dropout/mul_grad/ReshapeEtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/RealDiv_2*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџЈ@
б
Atraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Sum_1Sum?training/Adam/gradients/dropout_3/cond/dropout/truediv_grad/mulStraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv
К
Etraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Reshape_1ReshapeAtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Sum_1Ctraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@dropout_3/cond/dropout/truediv*
Tshape0*
_output_shapes
: 
З
5training/Adam/gradients/dropout_3/cond/mul_grad/ShapeShapedropout_3/cond/mul/Switch:1*
T0*%
_class
loc:@dropout_3/cond/mul*
out_type0*
_output_shapes
:
Ё
7training/Adam/gradients/dropout_3/cond/mul_grad/Shape_1Const*%
_class
loc:@dropout_3/cond/mul*
valueB *
dtype0*
_output_shapes
: 
В
Etraining/Adam/gradients/dropout_3/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_3/cond/mul_grad/Shape7training/Adam/gradients/dropout_3/cond/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@dropout_3/cond/mul
ј
3training/Adam/gradients/dropout_3/cond/mul_grad/MulMulCtraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Reshapedropout_3/cond/mul/y*
T0*%
_class
loc:@dropout_3/cond/mul*1
_output_shapes
:џџџџџџџџџЈ@

3training/Adam/gradients/dropout_3/cond/mul_grad/SumSum3training/Adam/gradients/dropout_3/cond/mul_grad/MulEtraining/Adam/gradients/dropout_3/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@dropout_3/cond/mul*
_output_shapes
:

7training/Adam/gradients/dropout_3/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_3/cond/mul_grad/Sum5training/Adam/gradients/dropout_3/cond/mul_grad/Shape*
T0*%
_class
loc:@dropout_3/cond/mul*
Tshape0*1
_output_shapes
:џџџџџџџџџЈ@

5training/Adam/gradients/dropout_3/cond/mul_grad/Mul_1Muldropout_3/cond/mul/Switch:1Ctraining/Adam/gradients/dropout_3/cond/dropout/truediv_grad/Reshape*1
_output_shapes
:џџџџџџџџџЈ@*
T0*%
_class
loc:@dropout_3/cond/mul
Ѓ
5training/Adam/gradients/dropout_3/cond/mul_grad/Sum_1Sum5training/Adam/gradients/dropout_3/cond/mul_grad/Mul_1Gtraining/Adam/gradients/dropout_3/cond/mul_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@dropout_3/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0

9training/Adam/gradients/dropout_3/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_3/cond/mul_grad/Sum_17training/Adam/gradients/dropout_3/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_3/cond/mul*
Tshape0*
_output_shapes
: 
р
 training/Adam/gradients/Switch_1Switchleaky_re_lu_6/LeakyReludropout_3/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*N
_output_shapes<
::џџџџџџџџџЈ@:џџџџџџџџџЈ@
И
"training/Adam/gradients/Identity_1Identity training/Adam/gradients/Switch_1*1
_output_shapes
:џџџџџџџџџЈ@*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu
Ћ
training/Adam/gradients/Shape_2Shape training/Adam/gradients/Switch_1*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*
out_type0*
_output_shapes
:
Л
%training/Adam/gradients/zeros_1/ConstConst#^training/Adam/gradients/Identity_1**
_class 
loc:@leaky_re_lu_6/LeakyRelu*
valueB
 *    *
dtype0*
_output_shapes
: 
щ
training/Adam/gradients/zeros_1Filltraining/Adam/gradients/Shape_2%training/Adam/gradients/zeros_1/Const*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*

index_type0*1
_output_shapes
:џџџџџџџџџЈ@

@training/Adam/gradients/dropout_3/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_17training/Adam/gradients/dropout_3/cond/mul_grad/Reshape*3
_output_shapes!
:џџџџџџџџџЈ@: *
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*
N

training/Adam/gradients/AddNAddN>training/Adam/gradients/dropout_3/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_3/cond/mul/Switch_grad/cond_grad*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*
N*1
_output_shapes
:џџџџџџџџџЈ@
ћ
Btraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGradLeakyReluGradtraining/Adam/gradients/AddNconv2d_6/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*
alpha%>*1
_output_shapes
:џџџџџџџџџЈ@
э
9training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_6/BiasAdd*
data_formatNHWC*
_output_shapes
:@
о
8training/Adam/gradients/conv2d_6/convolution_grad/ShapeNShapeNleaky_re_lu_5/LeakyReluconv2d_6/kernel/read*
T0*'
_class
loc:@conv2d_6/convolution*
out_type0*
N* 
_output_shapes
::
У
Etraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_6/convolution_grad/ShapeNconv2d_6/kernel/readBtraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGrad*
paddingSAME*1
_output_shapes
:џџџџџџџџџЈ@*
	dilations
*
T0*'
_class
loc:@conv2d_6/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
П
Ftraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterleaky_re_lu_5/LeakyRelu:training/Adam/gradients/conv2d_6/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGrad*&
_output_shapes
:@@*
	dilations
*
T0*'
_class
loc:@conv2d_6/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Є
Btraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGradLeakyReluGradEtraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputconv2d_5/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_5/LeakyRelu*
alpha%>*1
_output_shapes
:џџџџџџџџџЈ@
э
9training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_5/BiasAdd*
data_formatNHWC*
_output_shapes
:@
ь
8training/Adam/gradients/conv2d_5/convolution_grad/ShapeNShapeN%up_sampling2d_1/ResizeNearestNeighborconv2d_5/kernel/read*'
_class
loc:@conv2d_5/convolution*
out_type0*
N* 
_output_shapes
::*
T0
Ф
Etraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_5/convolution_grad/ShapeNconv2d_5/kernel/readBtraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGrad*2
_output_shapes 
:џџџџџџџџџЈ*
	dilations
*
T0*'
_class
loc:@conv2d_5/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Ю
Ftraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_1/ResizeNearestNeighbor:training/Adam/gradients/conv2d_5/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGrad*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*'
_class
loc:@conv2d_5/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
ь
atraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*
valueB"    *
dtype0*
_output_shapes
:
Џ
\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradEtraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropInputatraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*2
_output_shapes 
:џџџџџџџџџ*
align_corners( *
T0

@training/Adam/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGraddropout_2/cond/Mergemax_pooling2d_2/MaxPool\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*2
_output_shapes 
:џџџџџџџџџЈ*
T0**
_class 
loc:@max_pooling2d_2/MaxPool*
strides
*
data_formatNHWC*
ksize
*
paddingSAME
І
;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradSwitch@training/Adam/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGraddropout_2/cond/pred_id*
T0**
_class 
loc:@max_pooling2d_2/MaxPool*P
_output_shapes>
<:џџџџџџџџџЈ:џџџџџџџџџЈ
т
 training/Adam/gradients/Switch_2Switchleaky_re_lu_4/LeakyReludropout_2/cond/pred_id**
_class 
loc:@leaky_re_lu_4/LeakyRelu*P
_output_shapes>
<:џџџџџџџџџЈ:џџџџџџџџџЈ*
T0
Л
"training/Adam/gradients/Identity_2Identity"training/Adam/gradients/Switch_2:1**
_class 
loc:@leaky_re_lu_4/LeakyRelu*2
_output_shapes 
:џџџџџџџџџЈ*
T0
­
training/Adam/gradients/Shape_3Shape"training/Adam/gradients/Switch_2:1**
_class 
loc:@leaky_re_lu_4/LeakyRelu*
out_type0*
_output_shapes
:*
T0
Л
%training/Adam/gradients/zeros_2/ConstConst#^training/Adam/gradients/Identity_2*
_output_shapes
: **
_class 
loc:@leaky_re_lu_4/LeakyRelu*
valueB
 *    *
dtype0
ъ
training/Adam/gradients/zeros_2Filltraining/Adam/gradients/Shape_3%training/Adam/gradients/zeros_2/Const*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*

index_type0*2
_output_shapes 
:џџџџџџџџџЈ

>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_2*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*
N*4
_output_shapes"
 :џџџџџџџџџЈ: 
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

;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1dropout_2/cond/dropout/Floor*2
_output_shapes 
:џџџџџџџџџЈ*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
Н
;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Р
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
Tshape0*2
_output_shapes 
:џџџџџџџџџЈ

=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Muldropout_2/cond/dropout/truediv=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1*-
_class#
!loc:@dropout_2/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџЈ*
T0
У
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Ц
Atraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџЈ*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
Tshape0
Ц
Atraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/ShapeShapedropout_2/cond/mul*
_output_shapes
:*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
out_type0
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

Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDivRealDiv?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshapedropout_2/cond/dropout/sub*2
_output_shapes 
:џџџџџџџџџЈ*
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
а
Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/ReshapeReshape?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/SumAtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Shape*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*
Tshape0*2
_output_shapes 
:џџџџџџџџџЈ
Ъ
?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/NegNegdropout_2/cond/mul*2
_output_shapes 
:џџџџџџџџџЈ*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv

Etraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_1RealDiv?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Negdropout_2/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџЈ
Ѓ
Etraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_2RealDivEtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_1dropout_2/cond/dropout/sub*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџЈ*
T0
О
?training/Adam/gradients/dropout_2/cond/dropout/truediv_grad/mulMul?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeEtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/RealDiv_2*
T0*1
_class'
%#loc:@dropout_2/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџЈ
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
5training/Adam/gradients/dropout_2/cond/mul_grad/ShapeShapedropout_2/cond/mul/Switch:1*
_output_shapes
:*
T0*%
_class
loc:@dropout_2/cond/mul*
out_type0
Ё
7training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1Const*%
_class
loc:@dropout_2/cond/mul*
valueB *
dtype0*
_output_shapes
: 
В
Etraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_2/cond/mul_grad/Shape7training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@dropout_2/cond/mul
љ
3training/Adam/gradients/dropout_2/cond/mul_grad/MulMulCtraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Reshapedropout_2/cond/mul/y*
T0*%
_class
loc:@dropout_2/cond/mul*2
_output_shapes 
:џџџџџџџџџЈ

3training/Adam/gradients/dropout_2/cond/mul_grad/SumSum3training/Adam/gradients/dropout_2/cond/mul_grad/MulEtraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@dropout_2/cond/mul
 
7training/Adam/gradients/dropout_2/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_2/cond/mul_grad/Sum5training/Adam/gradients/dropout_2/cond/mul_grad/Shape*2
_output_shapes 
:џџџџџџџџџЈ*
T0*%
_class
loc:@dropout_2/cond/mul*
Tshape0

5training/Adam/gradients/dropout_2/cond/mul_grad/Mul_1Muldropout_2/cond/mul/Switch:1Ctraining/Adam/gradients/dropout_2/cond/dropout/truediv_grad/Reshape*
T0*%
_class
loc:@dropout_2/cond/mul*2
_output_shapes 
:џџџџџџџџџЈ
Ѓ
5training/Adam/gradients/dropout_2/cond/mul_grad/Sum_1Sum5training/Adam/gradients/dropout_2/cond/mul_grad/Mul_1Gtraining/Adam/gradients/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@dropout_2/cond/mul*
_output_shapes
:

9training/Adam/gradients/dropout_2/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_2/cond/mul_grad/Sum_17training/Adam/gradients/dropout_2/cond/mul_grad/Shape_1*
_output_shapes
: *
T0*%
_class
loc:@dropout_2/cond/mul*
Tshape0
т
 training/Adam/gradients/Switch_3Switchleaky_re_lu_4/LeakyReludropout_2/cond/pred_id**
_class 
loc:@leaky_re_lu_4/LeakyRelu*P
_output_shapes>
<:џџџџџџџџџЈ:џџџџџџџџџЈ*
T0
Й
"training/Adam/gradients/Identity_3Identity training/Adam/gradients/Switch_3*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*2
_output_shapes 
:џџџџџџџџџЈ
Ћ
training/Adam/gradients/Shape_4Shape training/Adam/gradients/Switch_3*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*
out_type0*
_output_shapes
:
Л
%training/Adam/gradients/zeros_3/ConstConst#^training/Adam/gradients/Identity_3*
_output_shapes
: **
_class 
loc:@leaky_re_lu_4/LeakyRelu*
valueB
 *    *
dtype0
ъ
training/Adam/gradients/zeros_3Filltraining/Adam/gradients/Shape_4%training/Adam/gradients/zeros_3/Const*2
_output_shapes 
:џџџџџџџџџЈ*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*

index_type0

@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_37training/Adam/gradients/dropout_2/cond/mul_grad/Reshape**
_class 
loc:@leaky_re_lu_4/LeakyRelu*
N*4
_output_shapes"
 :џџџџџџџџџЈ: *
T0

training/Adam/gradients/AddN_1AddN>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_2/cond/mul/Switch_grad/cond_grad*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*
N*2
_output_shapes 
:џџџџџџџџџЈ
ў
Btraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGradLeakyReluGradtraining/Adam/gradients/AddN_1conv2d_4/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*
alpha%>*2
_output_shapes 
:џџџџџџџџџЈ
ю
9training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0*#
_class
loc:@conv2d_4/BiasAdd
о
8training/Adam/gradients/conv2d_4/convolution_grad/ShapeNShapeNleaky_re_lu_3/LeakyReluconv2d_4/kernel/read*
T0*'
_class
loc:@conv2d_4/convolution*
out_type0*
N* 
_output_shapes
::
Ф
Etraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_4/convolution_grad/ShapeNconv2d_4/kernel/readBtraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_4/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:џџџџџџџџџЈ
С
Ftraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterleaky_re_lu_3/LeakyRelu:training/Adam/gradients/conv2d_4/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGrad*(
_output_shapes
:*
	dilations
*
T0*'
_class
loc:@conv2d_4/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Ѕ
Btraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGradLeakyReluGradEtraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputconv2d_3/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_3/LeakyRelu*
alpha%>*2
_output_shapes 
:џџџџџџџџџЈ
ю
9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_3/BiasAdd*
data_formatNHWC*
_output_shapes	
:
о
8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNShapeNmax_pooling2d_1/MaxPoolconv2d_3/kernel/read*
T0*'
_class
loc:@conv2d_3/convolution*
out_type0*
N* 
_output_shapes
::
У
Etraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNconv2d_3/kernel/readBtraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGrad*1
_output_shapes
:џџџџџџџџџЈ@*
	dilations
*
T0*'
_class
loc:@conv2d_3/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Р
Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_1/MaxPool:training/Adam/gradients/conv2d_3/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGrad*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*'
_class
loc:@conv2d_3/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
џ
@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGraddropout_1/cond/Mergemax_pooling2d_1/MaxPoolEtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInput*
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
:џџџџџџџџџИа@
Є
;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradSwitch@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGraddropout_1/cond/pred_id*
T0**
_class 
loc:@max_pooling2d_1/MaxPool*N
_output_shapes<
::џџџџџџџџџИа@:џџџџџџџџџИа@
р
 training/Adam/gradients/Switch_4Switchleaky_re_lu_2/LeakyReludropout_1/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*N
_output_shapes<
::џџџџџџџџџИа@:џџџџџџџџџИа@
К
"training/Adam/gradients/Identity_4Identity"training/Adam/gradients/Switch_4:1*1
_output_shapes
:џџџџџџџџџИа@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu
­
training/Adam/gradients/Shape_5Shape"training/Adam/gradients/Switch_4:1*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
out_type0*
_output_shapes
:
Л
%training/Adam/gradients/zeros_4/ConstConst#^training/Adam/gradients/Identity_4**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
valueB
 *    *
dtype0*
_output_shapes
: 
щ
training/Adam/gradients/zeros_4Filltraining/Adam/gradients/Shape_5%training/Adam/gradients/zeros_4/Const*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*

index_type0*1
_output_shapes
:џџџџџџџџџИа@

>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_4*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџИа@: 
Ъ
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ShapeShapedropout_1/cond/dropout/truediv*
_output_shapes
:*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0
Ъ
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1Shapedropout_1/cond/dropout/Floor*
_output_shapes
:*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0
в
Mtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul

;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMul=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1dropout_1/cond/dropout/Floor*1
_output_shapes
:џџџџџџџџџИа@*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
Н
;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
П
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape*1
_output_shapes
:џџџџџџџџџИа@*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0

=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Muldropout_1/cond/dropout/truediv=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*1
_output_shapes
:џџџџџџџџџИа@
У
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
Х
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0*1
_output_shapes
:џџџџџџџџџИа@
Ц
Atraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/ShapeShapedropout_1/cond/mul*
_output_shapes
:*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
out_type0
Й
Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
: *1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
valueB 
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
:џџџџџџџџџИа@
б
?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/SumSumCtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDivQtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
:
Я
Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/ReshapeReshape?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/SumAtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Shape*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
Tshape0*1
_output_shapes
:џџџџџџџџџИа@*
T0
Щ
?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/NegNegdropout_1/cond/mul*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа@

Etraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1RealDiv?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Negdropout_1/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа@
Ђ
Etraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2RealDivEtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1dropout_1/cond/dropout/sub*1
_output_shapes
:џџџџџџџџџИа@*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
Н
?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/mulMul?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeEtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*1
_output_shapes
:џџџџџџџџџИа@
б
Atraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1Sum?training/Adam/gradients/dropout_1/cond/dropout/truediv_grad/mulStraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0
К
Etraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Reshape_1ReshapeAtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
Tshape0*
_output_shapes
: 
З
5training/Adam/gradients/dropout_1/cond/mul_grad/ShapeShapedropout_1/cond/mul/Switch:1*
T0*%
_class
loc:@dropout_1/cond/mul*
out_type0*
_output_shapes
:
Ё
7training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1Const*%
_class
loc:@dropout_1/cond/mul*
valueB *
dtype0*
_output_shapes
: 
В
Etraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5training/Adam/gradients/dropout_1/cond/mul_grad/Shape7training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_1/cond/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ј
3training/Adam/gradients/dropout_1/cond/mul_grad/MulMulCtraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Reshapedropout_1/cond/mul/y*%
_class
loc:@dropout_1/cond/mul*1
_output_shapes
:џџџџџџџџџИа@*
T0

3training/Adam/gradients/dropout_1/cond/mul_grad/SumSum3training/Adam/gradients/dropout_1/cond/mul_grad/MulEtraining/Adam/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@dropout_1/cond/mul

7training/Adam/gradients/dropout_1/cond/mul_grad/ReshapeReshape3training/Adam/gradients/dropout_1/cond/mul_grad/Sum5training/Adam/gradients/dropout_1/cond/mul_grad/Shape*
T0*%
_class
loc:@dropout_1/cond/mul*
Tshape0*1
_output_shapes
:џџџџџџџџџИа@

5training/Adam/gradients/dropout_1/cond/mul_grad/Mul_1Muldropout_1/cond/mul/Switch:1Ctraining/Adam/gradients/dropout_1/cond/dropout/truediv_grad/Reshape*1
_output_shapes
:џџџџџџџџџИа@*
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
9training/Adam/gradients/dropout_1/cond/mul_grad/Reshape_1Reshape5training/Adam/gradients/dropout_1/cond/mul_grad/Sum_17training/Adam/gradients/dropout_1/cond/mul_grad/Shape_1*
_output_shapes
: *
T0*%
_class
loc:@dropout_1/cond/mul*
Tshape0
р
 training/Adam/gradients/Switch_5Switchleaky_re_lu_2/LeakyReludropout_1/cond/pred_id*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*N
_output_shapes<
::џџџџџџџџџИа@:џџџџџџџџџИа@
И
"training/Adam/gradients/Identity_5Identity training/Adam/gradients/Switch_5*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*1
_output_shapes
:џџџџџџџџџИа@
Ћ
training/Adam/gradients/Shape_6Shape training/Adam/gradients/Switch_5*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
out_type0*
_output_shapes
:
Л
%training/Adam/gradients/zeros_5/ConstConst#^training/Adam/gradients/Identity_5**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
valueB
 *    *
dtype0*
_output_shapes
: 
щ
training/Adam/gradients/zeros_5Filltraining/Adam/gradients/Shape_6%training/Adam/gradients/zeros_5/Const*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*

index_type0*1
_output_shapes
:џџџџџџџџџИа@

@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_57training/Adam/gradients/dropout_1/cond/mul_grad/Reshape*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N*3
_output_shapes!
:џџџџџџџџџИа@: 

training/Adam/gradients/AddN_2AddN>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_grad@training/Adam/gradients/dropout_1/cond/mul/Switch_grad/cond_grad*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
N*1
_output_shapes
:џџџџџџџџџИа@
§
Btraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGradLeakyReluGradtraining/Adam/gradients/AddN_2conv2d_2/BiasAdd*
alpha%>*1
_output_shapes
:џџџџџџџџџИа@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu
э
9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_2/BiasAdd*
data_formatNHWC*
_output_shapes
:@
о
8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNShapeNleaky_re_lu_1/LeakyReluconv2d_2/kernel/read*
T0*'
_class
loc:@conv2d_2/convolution*
out_type0*
N* 
_output_shapes
::
У
Etraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNconv2d_2/kernel/readBtraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџИа@*
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution
П
Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterleaky_re_lu_1/LeakyRelu:training/Adam/gradients/conv2d_2/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*&
_output_shapes
:@@*
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Є
Btraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGradLeakyReluGradEtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputconv2d_1/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
alpha%>*1
_output_shapes
:џџџџџџџџџИа@
э
9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_1/BiasAdd*
data_formatNHWC*
_output_shapes
:@
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
Etraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNconv2d_1/kernel/readBtraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
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
:џџџџџџџџџИа
Ж
Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_1_input:training/Adam/gradients/conv2d_1/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*&
_output_shapes
:@*
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
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

DstT0*
_output_shapes
: *

SrcT0	*
Truncate( 
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
training/Adam/sub/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
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
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
_output_shapes
: *
T0
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
|
#training/Adam/zeros/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
^
training/Adam/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zerosFill#training/Adam/zeros/shape_as_tensortraining/Adam/zeros/Const*&
_output_shapes
:@*
T0*

index_type0

training/Adam/Variable
VariableV2*
dtype0*
	container *&
_output_shapes
:@*
shape:@*
shared_name 
й
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*&
_output_shapes
:@

training/Adam/Variable/readIdentitytraining/Adam/Variable*&
_output_shapes
:@*
T0*)
_class
loc:@training/Adam/Variable
b
training/Adam/zeros_1Const*
_output_shapes
:@*
valueB@*    *
dtype0

training/Adam/Variable_1
VariableV2*
shape:@*
shared_name *
dtype0*
	container *
_output_shapes
:@
е
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:@
~
%training/Adam/zeros_2/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
`
training/Adam/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Є
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*&
_output_shapes
:@@*
T0*

index_type0

training/Adam/Variable_2
VariableV2*
dtype0*
	container *&
_output_shapes
:@@*
shape:@@*
shared_name 
с
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*&
_output_shapes
:@@
Ё
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
T0*+
_class!
loc:@training/Adam/Variable_2*&
_output_shapes
:@@
b
training/Adam/zeros_3Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_3
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:@*
shape:@
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
valueB"      @      *
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
Ѕ
training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_4
VariableV2*
shared_name *
dtype0*
	container *'
_output_shapes
:@*
shape:@
т
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*'
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(
Ђ
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4*'
_output_shapes
:@
d
training/Adam/zeros_5Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_5
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
ж
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes	
:*
use_locking(

training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes	
:*
T0
~
%training/Adam/zeros_6/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
`
training/Adam/zeros_6/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
І
training/Adam/zeros_6Fill%training/Adam/zeros_6/shape_as_tensortraining/Adam/zeros_6/Const*
T0*

index_type0*(
_output_shapes
:
 
training/Adam/Variable_6
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:*
shape:
у
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Ѓ
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
T0*+
_class!
loc:@training/Adam/Variable_6*(
_output_shapes
:
d
training/Adam/zeros_7Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_7
VariableV2*
	container *
_output_shapes	
:*
shape:*
shared_name *
dtype0
ж
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes	
:
~
%training/Adam/zeros_8/shape_as_tensorConst*
_output_shapes
:*%
valueB"         @   *
dtype0
`
training/Adam/zeros_8/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ѕ
training/Adam/zeros_8Fill%training/Adam/zeros_8/shape_as_tensortraining/Adam/zeros_8/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_8
VariableV2*
dtype0*
	container *'
_output_shapes
:@*
shape:@*
shared_name 
т
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*'
_output_shapes
:@*
use_locking(
Ђ
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
T0*+
_class!
loc:@training/Adam/Variable_8*'
_output_shapes
:@
b
training/Adam/zeros_9Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_9
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:@*
shape:@
е
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
:@

&training/Adam/zeros_10/shape_as_tensorConst*%
valueB"      @   @   *
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
:@@

training/Adam/Variable_10
VariableV2*
dtype0*
	container *&
_output_shapes
:@@*
shape:@@*
shared_name 
х
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*&
_output_shapes
:@@
Є
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*&
_output_shapes
:@@
c
training/Adam/zeros_11Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_11
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:@*
shape:@
й
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11

training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_11
{
training/Adam/zeros_12Const*
dtype0*&
_output_shapes
:@*%
valueB@*    

training/Adam/Variable_12
VariableV2*
dtype0*
	container *&
_output_shapes
:@*
shape:@*
shared_name 
х
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*&
_output_shapes
:@
Є
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*&
_output_shapes
:@*
T0
c
training/Adam/zeros_13Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_13
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(

training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_13

&training/Adam/zeros_14/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_14/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ї
training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*&
_output_shapes
:@*
T0*

index_type0

training/Adam/Variable_14
VariableV2*
dtype0*
	container *&
_output_shapes
:@*
shape:@*
shared_name 
х
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*&
_output_shapes
:@
Є
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*&
_output_shapes
:@
c
training/Adam/zeros_15Const*
_output_shapes
:@*
valueB@*    *
dtype0

training/Adam/Variable_15
VariableV2*
shape:@*
shared_name *
dtype0*
	container *
_output_shapes
:@
й
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
:@*
T0

&training/Adam/zeros_16/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_16/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ї
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*
T0*

index_type0*&
_output_shapes
:@@

training/Adam/Variable_16
VariableV2*
shape:@@*
shared_name *
dtype0*
	container *&
_output_shapes
:@@
х
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*&
_output_shapes
:@@*
use_locking(
Є
training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*&
_output_shapes
:@@*
T0*,
_class"
 loc:@training/Adam/Variable_16
c
training/Adam/zeros_17Const*
dtype0*
_output_shapes
:@*
valueB@*    

training/Adam/Variable_17
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
й
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:@*
use_locking(

training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
_output_shapes
:@

&training/Adam/zeros_18/shape_as_tensorConst*%
valueB"      @      *
dtype0*
_output_shapes
:
a
training/Adam/zeros_18/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ј
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_18
VariableV2*
dtype0*
	container *'
_output_shapes
:@*
shape:@*
shared_name 
ц
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*'
_output_shapes
:@*
use_locking(
Ѕ
training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*
T0*,
_class"
 loc:@training/Adam/Variable_18*'
_output_shapes
:@
e
training/Adam/zeros_19Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_19
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
к
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes	
:*
use_locking(

training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
_output_shapes	
:

&training/Adam/zeros_20/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_20/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Љ
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*(
_output_shapes
:*
T0*

index_type0
Ё
training/Adam/Variable_20
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
ч
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(*(
_output_shapes
:
І
training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
T0*,
_class"
 loc:@training/Adam/Variable_20*(
_output_shapes
:
e
training/Adam/zeros_21Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_21
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
к
 training/Adam/Variable_21/AssignAssigntraining/Adam/Variable_21training/Adam/zeros_21*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*
T0*,
_class"
 loc:@training/Adam/Variable_21*
_output_shapes	
:

&training/Adam/zeros_22/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"         @   
a
training/Adam/zeros_22/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ј
training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_22
VariableV2*
	container *'
_output_shapes
:@*
shape:@*
shared_name *
dtype0
ц
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*'
_output_shapes
:@
Ѕ
training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*
T0*,
_class"
 loc:@training/Adam/Variable_22*'
_output_shapes
:@
c
training/Adam/zeros_23Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_23
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
й
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(

training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*
T0*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes
:@

&training/Adam/zeros_24/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_24/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ї
training/Adam/zeros_24Fill&training/Adam/zeros_24/shape_as_tensortraining/Adam/zeros_24/Const*

index_type0*&
_output_shapes
:@@*
T0

training/Adam/Variable_24
VariableV2*
	container *&
_output_shapes
:@@*
shape:@@*
shared_name *
dtype0
х
 training/Adam/Variable_24/AssignAssigntraining/Adam/Variable_24training/Adam/zeros_24*
T0*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(*&
_output_shapes
:@@*
use_locking(
Є
training/Adam/Variable_24/readIdentitytraining/Adam/Variable_24*,
_class"
 loc:@training/Adam/Variable_24*&
_output_shapes
:@@*
T0
c
training/Adam/zeros_25Const*
_output_shapes
:@*
valueB@*    *
dtype0

training/Adam/Variable_25
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:@*
shape:@
й
 training/Adam/Variable_25/AssignAssigntraining/Adam/Variable_25training/Adam/zeros_25*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_25/readIdentitytraining/Adam/Variable_25*
T0*,
_class"
 loc:@training/Adam/Variable_25*
_output_shapes
:@
{
training/Adam/zeros_26Const*%
valueB@*    *
dtype0*&
_output_shapes
:@

training/Adam/Variable_26
VariableV2*
	container *&
_output_shapes
:@*
shape:@*
shared_name *
dtype0
х
 training/Adam/Variable_26/AssignAssigntraining/Adam/Variable_26training/Adam/zeros_26*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_26*
validate_shape(*&
_output_shapes
:@
Є
training/Adam/Variable_26/readIdentitytraining/Adam/Variable_26*
T0*,
_class"
 loc:@training/Adam/Variable_26*&
_output_shapes
:@
c
training/Adam/zeros_27Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_27
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
й
 training/Adam/Variable_27/AssignAssigntraining/Adam/Variable_27training/Adam/zeros_27*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_27*
validate_shape(

training/Adam/Variable_27/readIdentitytraining/Adam/Variable_27*
T0*,
_class"
 loc:@training/Adam/Variable_27*
_output_shapes
:
p
&training/Adam/zeros_28/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_28/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_28Fill&training/Adam/zeros_28/shape_as_tensortraining/Adam/zeros_28/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_28
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_28/AssignAssigntraining/Adam/Variable_28training/Adam/zeros_28*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_28*
validate_shape(*
_output_shapes
:

training/Adam/Variable_28/readIdentitytraining/Adam/Variable_28*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_28
p
&training/Adam/zeros_29/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_29/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_29Fill&training/Adam/zeros_29/shape_as_tensortraining/Adam/zeros_29/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_29
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
й
 training/Adam/Variable_29/AssignAssigntraining/Adam/Variable_29training/Adam/zeros_29*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_29

training/Adam/Variable_29/readIdentitytraining/Adam/Variable_29*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_29
p
&training/Adam/zeros_30/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_30/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_30Fill&training/Adam/zeros_30/shape_as_tensortraining/Adam/zeros_30/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_30
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
й
 training/Adam/Variable_30/AssignAssigntraining/Adam/Variable_30training/Adam/zeros_30*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_30*
validate_shape(

training/Adam/Variable_30/readIdentitytraining/Adam/Variable_30*,
_class"
 loc:@training/Adam/Variable_30*
_output_shapes
:*
T0
p
&training/Adam/zeros_31/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_31/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_31Fill&training/Adam/zeros_31/shape_as_tensortraining/Adam/zeros_31/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_31
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
й
 training/Adam/Variable_31/AssignAssigntraining/Adam/Variable_31training/Adam/zeros_31*
T0*,
_class"
 loc:@training/Adam/Variable_31*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_31/readIdentitytraining/Adam/Variable_31*,
_class"
 loc:@training/Adam/Variable_31*
_output_shapes
:*
T0
p
&training/Adam/zeros_32/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_32/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_32Fill&training/Adam/zeros_32/shape_as_tensortraining/Adam/zeros_32/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_32
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
й
 training/Adam/Variable_32/AssignAssigntraining/Adam/Variable_32training/Adam/zeros_32*
T0*,
_class"
 loc:@training/Adam/Variable_32*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_32/readIdentitytraining/Adam/Variable_32*
T0*,
_class"
 loc:@training/Adam/Variable_32*
_output_shapes
:
p
&training/Adam/zeros_33/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_33/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_33Fill&training/Adam/zeros_33/shape_as_tensortraining/Adam/zeros_33/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_33
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
й
 training/Adam/Variable_33/AssignAssigntraining/Adam/Variable_33training/Adam/zeros_33*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_33

training/Adam/Variable_33/readIdentitytraining/Adam/Variable_33*,
_class"
 loc:@training/Adam/Variable_33*
_output_shapes
:*
T0
p
&training/Adam/zeros_34/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_34/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_34Fill&training/Adam/zeros_34/shape_as_tensortraining/Adam/zeros_34/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_34
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
й
 training/Adam/Variable_34/AssignAssigntraining/Adam/Variable_34training/Adam/zeros_34*,
_class"
 loc:@training/Adam/Variable_34*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_34/readIdentitytraining/Adam/Variable_34*
T0*,
_class"
 loc:@training/Adam/Variable_34*
_output_shapes
:
p
&training/Adam/zeros_35/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_35/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_35Fill&training/Adam/zeros_35/shape_as_tensortraining/Adam/zeros_35/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_35
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
й
 training/Adam/Variable_35/AssignAssigntraining/Adam/Variable_35training/Adam/zeros_35*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_35

training/Adam/Variable_35/readIdentitytraining/Adam/Variable_35*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_35
p
&training/Adam/zeros_36/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_36/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_36Fill&training/Adam/zeros_36/shape_as_tensortraining/Adam/zeros_36/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_36
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
й
 training/Adam/Variable_36/AssignAssigntraining/Adam/Variable_36training/Adam/zeros_36*,
_class"
 loc:@training/Adam/Variable_36*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_36/readIdentitytraining/Adam/Variable_36*
T0*,
_class"
 loc:@training/Adam/Variable_36*
_output_shapes
:
p
&training/Adam/zeros_37/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_37/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_37Fill&training/Adam/zeros_37/shape_as_tensortraining/Adam/zeros_37/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_37
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
й
 training/Adam/Variable_37/AssignAssigntraining/Adam/Variable_37training/Adam/zeros_37*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_37*
validate_shape(

training/Adam/Variable_37/readIdentitytraining/Adam/Variable_37*
T0*,
_class"
 loc:@training/Adam/Variable_37*
_output_shapes
:
p
&training/Adam/zeros_38/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_38/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_38Fill&training/Adam/zeros_38/shape_as_tensortraining/Adam/zeros_38/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_38
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
й
 training/Adam/Variable_38/AssignAssigntraining/Adam/Variable_38training/Adam/zeros_38*
T0*,
_class"
 loc:@training/Adam/Variable_38*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_38/readIdentitytraining/Adam/Variable_38*
T0*,
_class"
 loc:@training/Adam/Variable_38*
_output_shapes
:
p
&training/Adam/zeros_39/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_39/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_39Fill&training/Adam/zeros_39/shape_as_tensortraining/Adam/zeros_39/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_39
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
й
 training/Adam/Variable_39/AssignAssigntraining/Adam/Variable_39training/Adam/zeros_39*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_39*
validate_shape(*
_output_shapes
:

training/Adam/Variable_39/readIdentitytraining/Adam/Variable_39*
T0*,
_class"
 loc:@training/Adam/Variable_39*
_output_shapes
:
p
&training/Adam/zeros_40/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_40/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_40Fill&training/Adam/zeros_40/shape_as_tensortraining/Adam/zeros_40/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_40
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
й
 training/Adam/Variable_40/AssignAssigntraining/Adam/Variable_40training/Adam/zeros_40*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_40*
validate_shape(*
_output_shapes
:

training/Adam/Variable_40/readIdentitytraining/Adam/Variable_40*
T0*,
_class"
 loc:@training/Adam/Variable_40*
_output_shapes
:
p
&training/Adam/zeros_41/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_41/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_41Fill&training/Adam/zeros_41/shape_as_tensortraining/Adam/zeros_41/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_41
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
й
 training/Adam/Variable_41/AssignAssigntraining/Adam/Variable_41training/Adam/zeros_41*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_41

training/Adam/Variable_41/readIdentitytraining/Adam/Variable_41*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_41
z
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
T0*&
_output_shapes
:@
Z
training/Adam/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
_output_shapes
: *
T0
Ј
training/Adam/mul_2Multraining/Adam/sub_2Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
u
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*&
_output_shapes
:@
}
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0*&
_output_shapes
:@
Z
training/Adam/sub_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/SquareSquareFtraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
v
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*&
_output_shapes
:@*
T0
u
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*&
_output_shapes
:@
s
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*&
_output_shapes
:@
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
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*&
_output_shapes
:@*
T0

training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
T0*&
_output_shapes
:@
l
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*&
_output_shapes
:@*
T0
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
:@
}
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0*&
_output_shapes
:@
z
training/Adam/sub_4Subconv2d_1/kernel/readtraining/Adam/truediv_1*
T0*&
_output_shapes
:@
а
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*)
_class
loc:@training/Adam/Variable*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
и
training/Adam/Assign_1Assigntraining/Adam/Variable_14training/Adam/add_2*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
Ф
training/Adam/Assign_2Assignconv2d_1/kerneltraining/Adam/sub_4*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@*
use_locking(
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
T0*
_output_shapes
:@
Z
training/Adam/sub_5/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_7Multraining/Adam/sub_59training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
:@
q
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_15/read*
T0*
_output_shapes
:@
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
:@*
T0
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
_output_shapes
:@*
T0
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
:@
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
:@
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
:@

training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
T0*
_output_shapes
:@
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
_output_shapes
:@*
T0
Z
training/Adam/add_6/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes
:@
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
_output_shapes
:@*
T0
l
training/Adam/sub_7Subconv2d_1/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes
:@
Ъ
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:@*
use_locking(
Ь
training/Adam/Assign_4Assigntraining/Adam/Variable_15training/Adam/add_5*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:@
Д
training/Adam/Assign_5Assignconv2d_1/biastraining/Adam/sub_7* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
}
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0*&
_output_shapes
:@@
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
training/Adam/mul_12Multraining/Adam/sub_8Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@@
w
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*&
_output_shapes
:@@
~
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_16/read*&
_output_shapes
:@@*
T0
Z
training/Adam/sub_9/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_2SquareFtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
y
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*&
_output_shapes
:@@*
T0
w
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*&
_output_shapes
:@@*
T0
t
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*&
_output_shapes
:@@*
T0
Z
training/Adam/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_7Const*
dtype0*
_output_shapes
: *
valueB
 *  

%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*
T0*&
_output_shapes
:@@

training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
T0*&
_output_shapes
:@@
l
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*&
_output_shapes
:@@*
T0
Z
training/Adam/add_9/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
x
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0*&
_output_shapes
:@@
~
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*&
_output_shapes
:@@*
T0
{
training/Adam/sub_10Subconv2d_2/kernel/readtraining/Adam/truediv_3*
T0*&
_output_shapes
:@@
ж
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*&
_output_shapes
:@@*
use_locking(
и
training/Adam/Assign_7Assigntraining/Adam/Variable_16training/Adam/add_8*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*&
_output_shapes
:@@
Х
training/Adam/Assign_8Assignconv2d_2/kerneltraining/Adam/sub_10*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:@@
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
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
_output_shapes
:@*
T0
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_17/read*
T0*
_output_shapes
:@
[
training/Adam/sub_12/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
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
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
_output_shapes
:@*
T0
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
training/Adam/Const_9Const*
dtype0*
_output_shapes
: *
valueB
 *  
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
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
_output_shapes
:@*
T0
[
training/Adam/add_12/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes
:@
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
Ю
training/Adam/Assign_10Assigntraining/Adam/Variable_17training/Adam/add_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:@
Ж
training/Adam/Assign_11Assignconv2d_2/biastraining/Adam/sub_13*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
~
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*'
_output_shapes
:@*
T0
[
training/Adam/sub_14/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ћ
training/Adam/mul_22Multraining/Adam/sub_14Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
y
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*'
_output_shapes
:@*
T0

training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_18/read*'
_output_shapes
:@*
T0
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

training/Adam/Square_4SquareFtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
{
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*'
_output_shapes
:@
y
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*'
_output_shapes
:@
v
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*'
_output_shapes
:@
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

%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*'
_output_shapes
:@*
T0

training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
T0*'
_output_shapes
:@
m
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*'
_output_shapes
:@
[
training/Adam/add_15/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
{
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*'
_output_shapes
:@

training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*'
_output_shapes
:@*
T0
|
training/Adam/sub_16Subconv2d_3/kernel/readtraining/Adam/truediv_5*
T0*'
_output_shapes
:@
й
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*'
_output_shapes
:@*
use_locking(
л
training/Adam/Assign_13Assigntraining/Adam/Variable_18training/Adam/add_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*'
_output_shapes
:@
Ч
training/Adam/Assign_14Assignconv2d_3/kerneltraining/Adam/sub_16*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(*'
_output_shapes
:@
r
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
T0*
_output_shapes	
:
[
training/Adam/sub_17/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_27Multraining/Adam/sub_179training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
_output_shapes	
:*
T0
s
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_19/read*
T0*
_output_shapes	
:
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

training/Adam/Square_5Square9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
_output_shapes	
:*
T0
m
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
_output_shapes	
:*
T0
j
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes	
:
[
training/Adam/Const_12Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_13Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
T0*
_output_shapes	
:

training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
T0*
_output_shapes	
:
a
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes	
:
[
training/Adam/add_18/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
_output_shapes	
:*
T0
t
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0*
_output_shapes	
:
n
training/Adam/sub_19Subconv2d_3/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes	
:
Э
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Я
training/Adam/Assign_16Assigntraining/Adam/Variable_19training/Adam/add_17*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(
З
training/Adam/Assign_17Assignconv2d_3/biastraining/Adam/sub_19*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*(
_output_shapes
:
[
training/Adam/sub_20/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ќ
training/Adam/mul_32Multraining/Adam/sub_20Ftraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
z
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*(
_output_shapes
:

training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_20/read*
T0*(
_output_shapes
:
[
training/Adam/sub_21/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_6SquareFtraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
|
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*(
_output_shapes
:*
T0
z
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*(
_output_shapes
:
w
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*(
_output_shapes
:
[
training/Adam/Const_14Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_15Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_15*(
_output_shapes
:*
T0

training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*(
_output_shapes
:*
T0
n
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0*(
_output_shapes
:
[
training/Adam/add_21/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
|
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*(
_output_shapes
:*
T0

training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*(
_output_shapes
:
}
training/Adam/sub_22Subconv2d_4/kernel/readtraining/Adam/truediv_7*
T0*(
_output_shapes
:
к
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6
м
training/Adam/Assign_19Assigntraining/Adam/Variable_20training/Adam/add_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(*(
_output_shapes
:
Ш
training/Adam/Assign_20Assignconv2d_4/kerneltraining/Adam/sub_22*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
r
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
_output_shapes	
:*
T0
[
training/Adam/sub_23/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_37Multraining/Adam/sub_239training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
T0*
_output_shapes	
:
s
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_21/read*
T0*
_output_shapes	
:
[
training/Adam/sub_24/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_7Square9training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes	
:
m
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes	
:
j
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
_output_shapes	
:*
T0
[
training/Adam/Const_16Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_17Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_17*
_output_shapes	
:*
T0

training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_16*
_output_shapes	
:*
T0
a
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes	
:
[
training/Adam/add_24/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
o
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
_output_shapes	
:*
T0
t
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
T0*
_output_shapes	
:
n
training/Adam/sub_25Subconv2d_4/bias/readtraining/Adam/truediv_8*
_output_shapes	
:*
T0
Э
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes	
:
Я
training/Adam/Assign_22Assigntraining/Adam/Variable_21training/Adam/add_23*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21
З
training/Adam/Assign_23Assignconv2d_4/biastraining/Adam/sub_25*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(
~
training/Adam/mul_41MulAdam/beta_1/readtraining/Adam/Variable_8/read*
T0*'
_output_shapes
:@
[
training/Adam/sub_26/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_26Subtraining/Adam/sub_26/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ћ
training/Adam/mul_42Multraining/Adam/sub_26Ftraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
y
training/Adam/add_25Addtraining/Adam/mul_41training/Adam/mul_42*'
_output_shapes
:@*
T0

training/Adam/mul_43MulAdam/beta_2/readtraining/Adam/Variable_22/read*
T0*'
_output_shapes
:@
[
training/Adam/sub_27/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_27Subtraining/Adam/sub_27/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_8SquareFtraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
{
training/Adam/mul_44Multraining/Adam/sub_27training/Adam/Square_8*
T0*'
_output_shapes
:@
y
training/Adam/add_26Addtraining/Adam/mul_43training/Adam/mul_44*'
_output_shapes
:@*
T0
v
training/Adam/mul_45Multraining/Adam/multraining/Adam/add_25*
T0*'
_output_shapes
:@
[
training/Adam/Const_18Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_19Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_9/MinimumMinimumtraining/Adam/add_26training/Adam/Const_19*'
_output_shapes
:@*
T0

training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_18*
T0*'
_output_shapes
:@
m
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
T0*'
_output_shapes
:@
[
training/Adam/add_27/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
{
training/Adam/add_27Addtraining/Adam/Sqrt_9training/Adam/add_27/y*'
_output_shapes
:@*
T0

training/Adam/truediv_9RealDivtraining/Adam/mul_45training/Adam/add_27*
T0*'
_output_shapes
:@
|
training/Adam/sub_28Subconv2d_5/kernel/readtraining/Adam/truediv_9*
T0*'
_output_shapes
:@
й
training/Adam/Assign_24Assigntraining/Adam/Variable_8training/Adam/add_25*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*'
_output_shapes
:@*
use_locking(
л
training/Adam/Assign_25Assigntraining/Adam/Variable_22training/Adam/add_26*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*'
_output_shapes
:@
Ч
training/Adam/Assign_26Assignconv2d_5/kerneltraining/Adam/sub_28*'
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel*
validate_shape(
q
training/Adam/mul_46MulAdam/beta_1/readtraining/Adam/Variable_9/read*
T0*
_output_shapes
:@
[
training/Adam/sub_29/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_29Subtraining/Adam/sub_29/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_47Multraining/Adam/sub_299training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
l
training/Adam/add_28Addtraining/Adam/mul_46training/Adam/mul_47*
T0*
_output_shapes
:@
r
training/Adam/mul_48MulAdam/beta_2/readtraining/Adam/Variable_23/read*
T0*
_output_shapes
:@
[
training/Adam/sub_30/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_30Subtraining/Adam/sub_30/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_9Square9training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
n
training/Adam/mul_49Multraining/Adam/sub_30training/Adam/Square_9*
_output_shapes
:@*
T0
l
training/Adam/add_29Addtraining/Adam/mul_48training/Adam/mul_49*
_output_shapes
:@*
T0
i
training/Adam/mul_50Multraining/Adam/multraining/Adam/add_28*
T0*
_output_shapes
:@
[
training/Adam/Const_20Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_21Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_10/MinimumMinimumtraining/Adam/add_29training/Adam/Const_21*
T0*
_output_shapes
:@

training/Adam/clip_by_value_10Maximum&training/Adam/clip_by_value_10/Minimumtraining/Adam/Const_20*
_output_shapes
:@*
T0
b
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
T0*
_output_shapes
:@
[
training/Adam/add_30/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_30Addtraining/Adam/Sqrt_10training/Adam/add_30/y*
_output_shapes
:@*
T0
t
training/Adam/truediv_10RealDivtraining/Adam/mul_50training/Adam/add_30*
_output_shapes
:@*
T0
n
training/Adam/sub_31Subconv2d_5/bias/readtraining/Adam/truediv_10*
_output_shapes
:@*
T0
Ь
training/Adam/Assign_27Assigntraining/Adam/Variable_9training/Adam/add_28*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:@
Ю
training/Adam/Assign_28Assigntraining/Adam/Variable_23training/Adam/add_29*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(
Ж
training/Adam/Assign_29Assignconv2d_5/biastraining/Adam/sub_31*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias
~
training/Adam/mul_51MulAdam/beta_1/readtraining/Adam/Variable_10/read*&
_output_shapes
:@@*
T0
[
training/Adam/sub_32/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_32Subtraining/Adam/sub_32/xAdam/beta_1/read*
T0*
_output_shapes
: 
Њ
training/Adam/mul_52Multraining/Adam/sub_32Ftraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
x
training/Adam/add_31Addtraining/Adam/mul_51training/Adam/mul_52*&
_output_shapes
:@@*
T0
~
training/Adam/mul_53MulAdam/beta_2/readtraining/Adam/Variable_24/read*
T0*&
_output_shapes
:@@
[
training/Adam/sub_33/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_33Subtraining/Adam/sub_33/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_10SquareFtraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
{
training/Adam/mul_54Multraining/Adam/sub_33training/Adam/Square_10*
T0*&
_output_shapes
:@@
x
training/Adam/add_32Addtraining/Adam/mul_53training/Adam/mul_54*&
_output_shapes
:@@*
T0
u
training/Adam/mul_55Multraining/Adam/multraining/Adam/add_31*
T0*&
_output_shapes
:@@
[
training/Adam/Const_22Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_23Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_11/MinimumMinimumtraining/Adam/add_32training/Adam/Const_23*
T0*&
_output_shapes
:@@

training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_22*&
_output_shapes
:@@*
T0
n
training/Adam/Sqrt_11Sqrttraining/Adam/clip_by_value_11*
T0*&
_output_shapes
:@@
[
training/Adam/add_33/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
{
training/Adam/add_33Addtraining/Adam/Sqrt_11training/Adam/add_33/y*&
_output_shapes
:@@*
T0

training/Adam/truediv_11RealDivtraining/Adam/mul_55training/Adam/add_33*
T0*&
_output_shapes
:@@
|
training/Adam/sub_34Subconv2d_6/kernel/readtraining/Adam/truediv_11*
T0*&
_output_shapes
:@@
к
training/Adam/Assign_30Assigntraining/Adam/Variable_10training/Adam/add_31*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*&
_output_shapes
:@@
к
training/Adam/Assign_31Assigntraining/Adam/Variable_24training/Adam/add_32*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(*&
_output_shapes
:@@
Ц
training/Adam/Assign_32Assignconv2d_6/kerneltraining/Adam/sub_34*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*&
_output_shapes
:@@*
use_locking(
r
training/Adam/mul_56MulAdam/beta_1/readtraining/Adam/Variable_11/read*
T0*
_output_shapes
:@
[
training/Adam/sub_35/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_35Subtraining/Adam/sub_35/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_57Multraining/Adam/sub_359training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
l
training/Adam/add_34Addtraining/Adam/mul_56training/Adam/mul_57*
T0*
_output_shapes
:@
r
training/Adam/mul_58MulAdam/beta_2/readtraining/Adam/Variable_25/read*
T0*
_output_shapes
:@
[
training/Adam/sub_36/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_36Subtraining/Adam/sub_36/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_11Square9training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
o
training/Adam/mul_59Multraining/Adam/sub_36training/Adam/Square_11*
T0*
_output_shapes
:@
l
training/Adam/add_35Addtraining/Adam/mul_58training/Adam/mul_59*
T0*
_output_shapes
:@
i
training/Adam/mul_60Multraining/Adam/multraining/Adam/add_34*
T0*
_output_shapes
:@
[
training/Adam/Const_24Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_25Const*
dtype0*
_output_shapes
: *
valueB
 *  

&training/Adam/clip_by_value_12/MinimumMinimumtraining/Adam/add_35training/Adam/Const_25*
T0*
_output_shapes
:@

training/Adam/clip_by_value_12Maximum&training/Adam/clip_by_value_12/Minimumtraining/Adam/Const_24*
T0*
_output_shapes
:@
b
training/Adam/Sqrt_12Sqrttraining/Adam/clip_by_value_12*
T0*
_output_shapes
:@
[
training/Adam/add_36/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_36Addtraining/Adam/Sqrt_12training/Adam/add_36/y*
T0*
_output_shapes
:@
t
training/Adam/truediv_12RealDivtraining/Adam/mul_60training/Adam/add_36*
T0*
_output_shapes
:@
n
training/Adam/sub_37Subconv2d_6/bias/readtraining/Adam/truediv_12*
T0*
_output_shapes
:@
Ю
training/Adam/Assign_33Assigntraining/Adam/Variable_11training/Adam/add_34*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11
Ю
training/Adam/Assign_34Assigntraining/Adam/Variable_25training/Adam/add_35*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(*
_output_shapes
:@
Ж
training/Adam/Assign_35Assignconv2d_6/biastraining/Adam/sub_37*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@conv2d_6/bias
~
training/Adam/mul_61MulAdam/beta_1/readtraining/Adam/Variable_12/read*&
_output_shapes
:@*
T0
[
training/Adam/sub_38/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_38Subtraining/Adam/sub_38/xAdam/beta_1/read*
_output_shapes
: *
T0
Њ
training/Adam/mul_62Multraining/Adam/sub_38Ftraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@*
T0
x
training/Adam/add_37Addtraining/Adam/mul_61training/Adam/mul_62*
T0*&
_output_shapes
:@
~
training/Adam/mul_63MulAdam/beta_2/readtraining/Adam/Variable_26/read*
T0*&
_output_shapes
:@
[
training/Adam/sub_39/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_39Subtraining/Adam/sub_39/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_12SquareFtraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
{
training/Adam/mul_64Multraining/Adam/sub_39training/Adam/Square_12*&
_output_shapes
:@*
T0
x
training/Adam/add_38Addtraining/Adam/mul_63training/Adam/mul_64*&
_output_shapes
:@*
T0
u
training/Adam/mul_65Multraining/Adam/multraining/Adam/add_37*
T0*&
_output_shapes
:@
[
training/Adam/Const_26Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_27Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_13/MinimumMinimumtraining/Adam/add_38training/Adam/Const_27*
T0*&
_output_shapes
:@

training/Adam/clip_by_value_13Maximum&training/Adam/clip_by_value_13/Minimumtraining/Adam/Const_26*
T0*&
_output_shapes
:@
n
training/Adam/Sqrt_13Sqrttraining/Adam/clip_by_value_13*&
_output_shapes
:@*
T0
[
training/Adam/add_39/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
{
training/Adam/add_39Addtraining/Adam/Sqrt_13training/Adam/add_39/y*&
_output_shapes
:@*
T0

training/Adam/truediv_13RealDivtraining/Adam/mul_65training/Adam/add_39*&
_output_shapes
:@*
T0
|
training/Adam/sub_40Subconv2d_7/kernel/readtraining/Adam/truediv_13*&
_output_shapes
:@*
T0
к
training/Adam/Assign_36Assigntraining/Adam/Variable_12training/Adam/add_37*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
к
training/Adam/Assign_37Assigntraining/Adam/Variable_26training/Adam/add_38*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_26*
validate_shape(*&
_output_shapes
:@
Ц
training/Adam/Assign_38Assignconv2d_7/kerneltraining/Adam/sub_40*
use_locking(*
T0*"
_class
loc:@conv2d_7/kernel*
validate_shape(*&
_output_shapes
:@
r
training/Adam/mul_66MulAdam/beta_1/readtraining/Adam/Variable_13/read*
T0*
_output_shapes
:
[
training/Adam/sub_41/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_41Subtraining/Adam/sub_41/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_67Multraining/Adam/sub_419training/Adam/gradients/conv2d_7/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_40Addtraining/Adam/mul_66training/Adam/mul_67*
_output_shapes
:*
T0
r
training/Adam/mul_68MulAdam/beta_2/readtraining/Adam/Variable_27/read*
T0*
_output_shapes
:
[
training/Adam/sub_42/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_42Subtraining/Adam/sub_42/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_13Square9training/Adam/gradients/conv2d_7/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
o
training/Adam/mul_69Multraining/Adam/sub_42training/Adam/Square_13*
_output_shapes
:*
T0
l
training/Adam/add_41Addtraining/Adam/mul_68training/Adam/mul_69*
_output_shapes
:*
T0
i
training/Adam/mul_70Multraining/Adam/multraining/Adam/add_40*
_output_shapes
:*
T0
[
training/Adam/Const_28Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_29Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_14/MinimumMinimumtraining/Adam/add_41training/Adam/Const_29*
T0*
_output_shapes
:

training/Adam/clip_by_value_14Maximum&training/Adam/clip_by_value_14/Minimumtraining/Adam/Const_28*
T0*
_output_shapes
:
b
training/Adam/Sqrt_14Sqrttraining/Adam/clip_by_value_14*
T0*
_output_shapes
:
[
training/Adam/add_42/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
o
training/Adam/add_42Addtraining/Adam/Sqrt_14training/Adam/add_42/y*
T0*
_output_shapes
:
t
training/Adam/truediv_14RealDivtraining/Adam/mul_70training/Adam/add_42*
_output_shapes
:*
T0
n
training/Adam/sub_43Subconv2d_7/bias/readtraining/Adam/truediv_14*
T0*
_output_shapes
:
Ю
training/Adam/Assign_39Assigntraining/Adam/Variable_13training/Adam/add_40*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:
Ю
training/Adam/Assign_40Assigntraining/Adam/Variable_27training/Adam/add_41*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_27
Ж
training/Adam/Assign_41Assignconv2d_7/biastraining/Adam/sub_43*
use_locking(*
T0* 
_class
loc:@conv2d_7/bias*
validate_shape(*
_output_shapes
:
ј
training/group_depsNoOp	^loss/mul^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_24^training/Adam/Assign_25^training/Adam/Assign_26^training/Adam/Assign_27^training/Adam/Assign_28^training/Adam/Assign_29^training/Adam/Assign_3^training/Adam/Assign_30^training/Adam/Assign_31^training/Adam/Assign_32^training/Adam/Assign_33^training/Adam/Assign_34^training/Adam/Assign_35^training/Adam/Assign_36^training/Adam/Assign_37^training/Adam/Assign_38^training/Adam/Assign_39^training/Adam/Assign_4^training/Adam/Assign_40^training/Adam/Assign_41^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9


group_depsNoOp	^loss/mul

IsVariableInitializedIsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitializedconv2d_1/bias*
_output_shapes
: * 
_class
loc:@conv2d_1/bias*
dtype0

IsVariableInitialized_2IsVariableInitializedconv2d_2/kernel*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel*
dtype0

IsVariableInitialized_3IsVariableInitializedconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializedconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializedconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_6IsVariableInitializedconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_7IsVariableInitializedconv2d_4/bias* 
_class
loc:@conv2d_4/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializedconv2d_5/kernel*"
_class
loc:@conv2d_5/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_9IsVariableInitializedconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitializedconv2d_6/kernel*
_output_shapes
: *"
_class
loc:@conv2d_6/kernel*
dtype0

IsVariableInitialized_11IsVariableInitializedconv2d_6/bias*
_output_shapes
: * 
_class
loc:@conv2d_6/bias*
dtype0

IsVariableInitialized_12IsVariableInitializedconv2d_7/kernel*
_output_shapes
: *"
_class
loc:@conv2d_7/kernel*
dtype0

IsVariableInitialized_13IsVariableInitializedconv2d_7/bias*
_output_shapes
: * 
_class
loc:@conv2d_7/bias*
dtype0

IsVariableInitialized_14IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
{
IsVariableInitialized_15IsVariableInitializedAdam/lr*
dtype0*
_output_shapes
: *
_class
loc:@Adam/lr

IsVariableInitialized_16IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_17IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_18IsVariableInitialized
Adam/decay*
dtype0*
_output_shapes
: *
_class
loc:@Adam/decay

IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
dtype0*
_output_shapes
: 

IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0*
_output_shapes
: 

IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0*
_output_shapes
: 

IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
: 

IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_6*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_6*
dtype0

IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0*
_output_shapes
: 

IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes
: 

IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0*
_output_shapes
: 

IsVariableInitialized_29IsVariableInitializedtraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0*
_output_shapes
: 

IsVariableInitialized_30IsVariableInitializedtraining/Adam/Variable_11*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_11*
dtype0

IsVariableInitialized_31IsVariableInitializedtraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
: 

IsVariableInitialized_32IsVariableInitializedtraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
dtype0*
_output_shapes
: 

IsVariableInitialized_33IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes
: 

IsVariableInitialized_34IsVariableInitializedtraining/Adam/Variable_15*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_15*
dtype0

IsVariableInitialized_35IsVariableInitializedtraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0*
_output_shapes
: 

IsVariableInitialized_36IsVariableInitializedtraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
dtype0*
_output_shapes
: 

IsVariableInitialized_37IsVariableInitializedtraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
dtype0*
_output_shapes
: 

IsVariableInitialized_38IsVariableInitializedtraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
dtype0*
_output_shapes
: 

IsVariableInitialized_39IsVariableInitializedtraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0*
_output_shapes
: 

IsVariableInitialized_40IsVariableInitializedtraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
: 

IsVariableInitialized_41IsVariableInitializedtraining/Adam/Variable_22*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_22

IsVariableInitialized_42IsVariableInitializedtraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
dtype0*
_output_shapes
: 

IsVariableInitialized_43IsVariableInitializedtraining/Adam/Variable_24*,
_class"
 loc:@training/Adam/Variable_24*
dtype0*
_output_shapes
: 

IsVariableInitialized_44IsVariableInitializedtraining/Adam/Variable_25*,
_class"
 loc:@training/Adam/Variable_25*
dtype0*
_output_shapes
: 

IsVariableInitialized_45IsVariableInitializedtraining/Adam/Variable_26*,
_class"
 loc:@training/Adam/Variable_26*
dtype0*
_output_shapes
: 

IsVariableInitialized_46IsVariableInitializedtraining/Adam/Variable_27*,
_class"
 loc:@training/Adam/Variable_27*
dtype0*
_output_shapes
: 

IsVariableInitialized_47IsVariableInitializedtraining/Adam/Variable_28*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_28

IsVariableInitialized_48IsVariableInitializedtraining/Adam/Variable_29*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_29*
dtype0

IsVariableInitialized_49IsVariableInitializedtraining/Adam/Variable_30*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_30

IsVariableInitialized_50IsVariableInitializedtraining/Adam/Variable_31*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_31

IsVariableInitialized_51IsVariableInitializedtraining/Adam/Variable_32*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_32

IsVariableInitialized_52IsVariableInitializedtraining/Adam/Variable_33*,
_class"
 loc:@training/Adam/Variable_33*
dtype0*
_output_shapes
: 

IsVariableInitialized_53IsVariableInitializedtraining/Adam/Variable_34*,
_class"
 loc:@training/Adam/Variable_34*
dtype0*
_output_shapes
: 

IsVariableInitialized_54IsVariableInitializedtraining/Adam/Variable_35*,
_class"
 loc:@training/Adam/Variable_35*
dtype0*
_output_shapes
: 

IsVariableInitialized_55IsVariableInitializedtraining/Adam/Variable_36*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_36*
dtype0

IsVariableInitialized_56IsVariableInitializedtraining/Adam/Variable_37*,
_class"
 loc:@training/Adam/Variable_37*
dtype0*
_output_shapes
: 

IsVariableInitialized_57IsVariableInitializedtraining/Adam/Variable_38*,
_class"
 loc:@training/Adam/Variable_38*
dtype0*
_output_shapes
: 

IsVariableInitialized_58IsVariableInitializedtraining/Adam/Variable_39*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_39*
dtype0

IsVariableInitialized_59IsVariableInitializedtraining/Adam/Variable_40*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_40

IsVariableInitialized_60IsVariableInitializedtraining/Adam/Variable_41*,
_class"
 loc:@training/Adam/Variable_41*
dtype0*
_output_shapes
: 
і
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^conv2d_3/bias/Assign^conv2d_3/kernel/Assign^conv2d_4/bias/Assign^conv2d_4/kernel/Assign^conv2d_5/bias/Assign^conv2d_5/kernel/Assign^conv2d_6/bias/Assign^conv2d_6/kernel/Assign^conv2d_7/bias/Assign^conv2d_7/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign!^training/Adam/Variable_24/Assign!^training/Adam/Variable_25/Assign!^training/Adam/Variable_26/Assign!^training/Adam/Variable_27/Assign!^training/Adam/Variable_28/Assign!^training/Adam/Variable_29/Assign ^training/Adam/Variable_3/Assign!^training/Adam/Variable_30/Assign!^training/Adam/Variable_31/Assign!^training/Adam/Variable_32/Assign!^training/Adam/Variable_33/Assign!^training/Adam/Variable_34/Assign!^training/Adam/Variable_35/Assign!^training/Adam/Variable_36/Assign!^training/Adam/Variable_37/Assign!^training/Adam/Variable_38/Assign!^training/Adam/Variable_39/Assign ^training/Adam/Variable_4/Assign!^training/Adam/Variable_40/Assign!^training/Adam/Variable_41/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign""ш6
trainable_variablesа6Э6
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
`
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:02conv2d_4/random_uniform:08
Q
conv2d_4/bias:0conv2d_4/bias/Assignconv2d_4/bias/read:02conv2d_4/Const:08
`
conv2d_5/kernel:0conv2d_5/kernel/Assignconv2d_5/kernel/read:02conv2d_5/random_uniform:08
Q
conv2d_5/bias:0conv2d_5/bias/Assignconv2d_5/bias/read:02conv2d_5/Const:08
`
conv2d_6/kernel:0conv2d_6/kernel/Assignconv2d_6/kernel/read:02conv2d_6/random_uniform:08
Q
conv2d_6/bias:0conv2d_6/bias/Assignconv2d_6/bias/read:02conv2d_6/Const:08
`
conv2d_7/kernel:0conv2d_7/kernel/Assignconv2d_7/kernel/read:02conv2d_7/random_uniform:08
Q
conv2d_7/bias:0conv2d_7/bias/Assignconv2d_7/bias/read:02conv2d_7/Const:08
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
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign training/Adam/Variable_17/read:02training/Adam/zeros_17:08
}
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign training/Adam/Variable_18/read:02training/Adam/zeros_18:08
}
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign training/Adam/Variable_19/read:02training/Adam/zeros_19:08
}
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign training/Adam/Variable_20/read:02training/Adam/zeros_20:08
}
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign training/Adam/Variable_21/read:02training/Adam/zeros_21:08
}
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign training/Adam/Variable_22/read:02training/Adam/zeros_22:08
}
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08
}
training/Adam/Variable_24:0 training/Adam/Variable_24/Assign training/Adam/Variable_24/read:02training/Adam/zeros_24:08
}
training/Adam/Variable_25:0 training/Adam/Variable_25/Assign training/Adam/Variable_25/read:02training/Adam/zeros_25:08
}
training/Adam/Variable_26:0 training/Adam/Variable_26/Assign training/Adam/Variable_26/read:02training/Adam/zeros_26:08
}
training/Adam/Variable_27:0 training/Adam/Variable_27/Assign training/Adam/Variable_27/read:02training/Adam/zeros_27:08
}
training/Adam/Variable_28:0 training/Adam/Variable_28/Assign training/Adam/Variable_28/read:02training/Adam/zeros_28:08
}
training/Adam/Variable_29:0 training/Adam/Variable_29/Assign training/Adam/Variable_29/read:02training/Adam/zeros_29:08
}
training/Adam/Variable_30:0 training/Adam/Variable_30/Assign training/Adam/Variable_30/read:02training/Adam/zeros_30:08
}
training/Adam/Variable_31:0 training/Adam/Variable_31/Assign training/Adam/Variable_31/read:02training/Adam/zeros_31:08
}
training/Adam/Variable_32:0 training/Adam/Variable_32/Assign training/Adam/Variable_32/read:02training/Adam/zeros_32:08
}
training/Adam/Variable_33:0 training/Adam/Variable_33/Assign training/Adam/Variable_33/read:02training/Adam/zeros_33:08
}
training/Adam/Variable_34:0 training/Adam/Variable_34/Assign training/Adam/Variable_34/read:02training/Adam/zeros_34:08
}
training/Adam/Variable_35:0 training/Adam/Variable_35/Assign training/Adam/Variable_35/read:02training/Adam/zeros_35:08
}
training/Adam/Variable_36:0 training/Adam/Variable_36/Assign training/Adam/Variable_36/read:02training/Adam/zeros_36:08
}
training/Adam/Variable_37:0 training/Adam/Variable_37/Assign training/Adam/Variable_37/read:02training/Adam/zeros_37:08
}
training/Adam/Variable_38:0 training/Adam/Variable_38/Assign training/Adam/Variable_38/read:02training/Adam/zeros_38:08
}
training/Adam/Variable_39:0 training/Adam/Variable_39/Assign training/Adam/Variable_39/read:02training/Adam/zeros_39:08
}
training/Adam/Variable_40:0 training/Adam/Variable_40/Assign training/Adam/Variable_40/read:02training/Adam/zeros_40:08
}
training/Adam/Variable_41:0 training/Adam/Variable_41/Assign training/Adam/Variable_41/read:02training/Adam/zeros_41:08"Ш
cond_contextЗД
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
leaky_re_lu_2/LeakyRelu:08
leaky_re_lu_2/LeakyRelu:0dropout_1/cond/mul/Switch:14
dropout_1/cond/pred_id:0dropout_1/cond/pred_id:0
Ш
dropout_1/cond/cond_text_1dropout_1/cond/pred_id:0dropout_1/cond/switch_f:0*є
dropout_1/cond/Switch_1:0
dropout_1/cond/Switch_1:1
dropout_1/cond/pred_id:0
dropout_1/cond/switch_f:0
leaky_re_lu_2/LeakyRelu:04
dropout_1/cond/pred_id:0dropout_1/cond/pred_id:06
leaky_re_lu_2/LeakyRelu:0dropout_1/cond/Switch_1:0
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
leaky_re_lu_4/LeakyRelu:04
dropout_2/cond/pred_id:0dropout_2/cond/pred_id:08
leaky_re_lu_4/LeakyRelu:0dropout_2/cond/mul/Switch:1
Ш
dropout_2/cond/cond_text_1dropout_2/cond/pred_id:0dropout_2/cond/switch_f:0*є
dropout_2/cond/Switch_1:0
dropout_2/cond/Switch_1:1
dropout_2/cond/pred_id:0
dropout_2/cond/switch_f:0
leaky_re_lu_4/LeakyRelu:06
leaky_re_lu_4/LeakyRelu:0dropout_2/cond/Switch_1:04
dropout_2/cond/pred_id:0dropout_2/cond/pred_id:0
ю
dropout_3/cond/cond_textdropout_3/cond/pred_id:0dropout_3/cond/switch_t:0 *
dropout_3/cond/dropout/Floor:0
dropout_3/cond/dropout/Shape:0
dropout_3/cond/dropout/add:0
dropout_3/cond/dropout/mul:0
5dropout_3/cond/dropout/random_uniform/RandomUniform:0
+dropout_3/cond/dropout/random_uniform/max:0
+dropout_3/cond/dropout/random_uniform/min:0
+dropout_3/cond/dropout/random_uniform/mul:0
+dropout_3/cond/dropout/random_uniform/sub:0
'dropout_3/cond/dropout/random_uniform:0
dropout_3/cond/dropout/rate:0
dropout_3/cond/dropout/sub/x:0
dropout_3/cond/dropout/sub:0
 dropout_3/cond/dropout/truediv:0
dropout_3/cond/mul/Switch:1
dropout_3/cond/mul/y:0
dropout_3/cond/mul:0
dropout_3/cond/pred_id:0
dropout_3/cond/switch_t:0
leaky_re_lu_6/LeakyRelu:08
leaky_re_lu_6/LeakyRelu:0dropout_3/cond/mul/Switch:14
dropout_3/cond/pred_id:0dropout_3/cond/pred_id:0
Ш
dropout_3/cond/cond_text_1dropout_3/cond/pred_id:0dropout_3/cond/switch_f:0*є
dropout_3/cond/Switch_1:0
dropout_3/cond/Switch_1:1
dropout_3/cond/pred_id:0
dropout_3/cond/switch_f:0
leaky_re_lu_6/LeakyRelu:04
dropout_3/cond/pred_id:0dropout_3/cond/pred_id:06
leaky_re_lu_6/LeakyRelu:0dropout_3/cond/Switch_1:0"о6
	variablesа6Э6
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
`
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:02conv2d_4/random_uniform:08
Q
conv2d_4/bias:0conv2d_4/bias/Assignconv2d_4/bias/read:02conv2d_4/Const:08
`
conv2d_5/kernel:0conv2d_5/kernel/Assignconv2d_5/kernel/read:02conv2d_5/random_uniform:08
Q
conv2d_5/bias:0conv2d_5/bias/Assignconv2d_5/bias/read:02conv2d_5/Const:08
`
conv2d_6/kernel:0conv2d_6/kernel/Assignconv2d_6/kernel/read:02conv2d_6/random_uniform:08
Q
conv2d_6/bias:0conv2d_6/bias/Assignconv2d_6/bias/read:02conv2d_6/Const:08
`
conv2d_7/kernel:0conv2d_7/kernel/Assignconv2d_7/kernel/read:02conv2d_7/random_uniform:08
Q
conv2d_7/bias:0conv2d_7/bias/Assignconv2d_7/bias/read:02conv2d_7/Const:08
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
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign training/Adam/Variable_17/read:02training/Adam/zeros_17:08
}
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign training/Adam/Variable_18/read:02training/Adam/zeros_18:08
}
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign training/Adam/Variable_19/read:02training/Adam/zeros_19:08
}
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign training/Adam/Variable_20/read:02training/Adam/zeros_20:08
}
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign training/Adam/Variable_21/read:02training/Adam/zeros_21:08
}
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign training/Adam/Variable_22/read:02training/Adam/zeros_22:08
}
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08
}
training/Adam/Variable_24:0 training/Adam/Variable_24/Assign training/Adam/Variable_24/read:02training/Adam/zeros_24:08
}
training/Adam/Variable_25:0 training/Adam/Variable_25/Assign training/Adam/Variable_25/read:02training/Adam/zeros_25:08
}
training/Adam/Variable_26:0 training/Adam/Variable_26/Assign training/Adam/Variable_26/read:02training/Adam/zeros_26:08
}
training/Adam/Variable_27:0 training/Adam/Variable_27/Assign training/Adam/Variable_27/read:02training/Adam/zeros_27:08
}
training/Adam/Variable_28:0 training/Adam/Variable_28/Assign training/Adam/Variable_28/read:02training/Adam/zeros_28:08
}
training/Adam/Variable_29:0 training/Adam/Variable_29/Assign training/Adam/Variable_29/read:02training/Adam/zeros_29:08
}
training/Adam/Variable_30:0 training/Adam/Variable_30/Assign training/Adam/Variable_30/read:02training/Adam/zeros_30:08
}
training/Adam/Variable_31:0 training/Adam/Variable_31/Assign training/Adam/Variable_31/read:02training/Adam/zeros_31:08
}
training/Adam/Variable_32:0 training/Adam/Variable_32/Assign training/Adam/Variable_32/read:02training/Adam/zeros_32:08
}
training/Adam/Variable_33:0 training/Adam/Variable_33/Assign training/Adam/Variable_33/read:02training/Adam/zeros_33:08
}
training/Adam/Variable_34:0 training/Adam/Variable_34/Assign training/Adam/Variable_34/read:02training/Adam/zeros_34:08
}
training/Adam/Variable_35:0 training/Adam/Variable_35/Assign training/Adam/Variable_35/read:02training/Adam/zeros_35:08
}
training/Adam/Variable_36:0 training/Adam/Variable_36/Assign training/Adam/Variable_36/read:02training/Adam/zeros_36:08
}
training/Adam/Variable_37:0 training/Adam/Variable_37/Assign training/Adam/Variable_37/read:02training/Adam/zeros_37:08
}
training/Adam/Variable_38:0 training/Adam/Variable_38/Assign training/Adam/Variable_38/read:02training/Adam/zeros_38:08
}
training/Adam/Variable_39:0 training/Adam/Variable_39/Assign training/Adam/Variable_39/read:02training/Adam/zeros_39:08
}
training/Adam/Variable_40:0 training/Adam/Variable_40/Assign training/Adam/Variable_40/read:02training/Adam/zeros_40:08
}
training/Adam/Variable_41:0 training/Adam/Variable_41/Assign training/Adam/Variable_41/read:02training/Adam/zeros_41:08їozб       ЃK"	*љяЙ зA*

lossCь>?оЉ       ШС	&љяЙ зA*

val_loss>Т>ё^ч:       и-	ПXѓЙ зA*

lossHС>8pd&       йм2	РXѓЙ зA*

val_lossЖ>H§       и-	ЪДіЙ зA*

lossAоЃ>ыД№Z       йм2	ДіЙ зA*

val_lossМъ>з`ж       и-	1E0њЙ зA*

lossЈЗ>b)k       йм2	ІF0њЙ зA*

val_loss>РР       и-	'Щ§Й зA*

lossјI>ж~iУ       йм2	UЩ§Й зA*

val_lossќЁ>я2Фж       и-	КМGК зA*

loss4o>zon       йм2	гНGК зA*

val_loss\ч >ЖўP       и-		QДК зA*

lossћ>[є">       йм2	3SДК зA*

val_lossр->йO       и-	ёЙК зA*

lossЄ>gT       йм2	љКК зA*

val_lossц>яDѕЯ       и-	пўlК зA*

lossэ>Ѕ(р?       йм2	яџlК зA*

val_loss3gЃ> DЦ       и-	oдвК зA	*

loss=>7Х%       йм2	евК зA	*

val_loss§дЈ>Ђрм       и-	awК зA
*

lossM>Ш&       йм2	9dwК зA
*

val_losscrЇ>Z`яу       и-	ЏщК зA*

lossН>ЪЛУ       йм2	ПщК зA*

val_lossй>В       и-	?К зA*

lossж­>iш       йм2	?К зA*

val_loss­>]       и-	+пК зA*

lossmР>ЅVй       йм2	3рК зA*

val_lossЁ>щ
ќ       и-	лќтК зA*

lossd>ъеЄ       йм2	у§тК зA*

val_loss@>jbjь