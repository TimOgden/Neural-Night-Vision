       ŁK"	   H5×Abrain.Event:2ŁGr_     őąŔ 	§[1H5×A"ĺž	

input_inputPlaceholder*&
shape:˙˙˙˙˙˙˙˙˙¸Đ*
dtype0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
s
input/random_uniform/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
]
input/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *8JĚ˝
]
input/random_uniform/maxConst*
valueB
 *8JĚ=*
dtype0*
_output_shapes
: 
Ź
"input/random_uniform/RandomUniformRandomUniforminput/random_uniform/shape*
dtype0*&
_output_shapes
:@*
seed2÷Ú*
seedą˙ĺ)*
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
:@

input/random_uniformAddinput/random_uniform/mulinput/random_uniform/min*
T0*&
_output_shapes
:@

input/kernel
VariableV2*
shape:@*
shared_name *
dtype0*&
_output_shapes
:@*
	container 
ź
input/kernel/AssignAssigninput/kernelinput/random_uniform*
use_locking(*
T0*
_class
loc:@input/kernel*
validate_shape(*&
_output_shapes
:@
}
input/kernel/readIdentityinput/kernel*
T0*
_class
loc:@input/kernel*&
_output_shapes
:@
X
input/ConstConst*
dtype0*
_output_shapes
:@*
valueB@*    
v

input/bias
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
Ą
input/bias/AssignAssign
input/biasinput/Const*
use_locking(*
T0*
_class
loc:@input/bias*
validate_shape(*
_output_shapes
:@
k
input/bias/readIdentity
input/bias*
T0*
_class
loc:@input/bias*
_output_shapes
:@
p
input/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
ĺ
input/convolutionConv2Dinput_inputinput/kernel/read*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
	dilations
*
T0*
data_formatNHWC*
strides


input/BiasAddBiasAddinput/convolutioninput/bias/read*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0

leaky_re_lu_1/LeakyRelu	LeakyReluinput/BiasAdd*
T0*
alpha%>*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
v
conv2d_1/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
`
conv2d_1/random_uniform/minConst*
valueB
 *:Í˝*
dtype0*
_output_shapes
: 
`
conv2d_1/random_uniform/maxConst*
valueB
 *:Í=*
dtype0*
_output_shapes
: 
˛
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
T0*
dtype0*&
_output_shapes
:@@*
seed2ó¤*
seedą˙ĺ)
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
_output_shapes
: *
T0

conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*
T0*&
_output_shapes
:@@

conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*
T0*&
_output_shapes
:@@

conv2d_1/kernel
VariableV2*
dtype0*&
_output_shapes
:@@*
	container *
shape:@@*
shared_name 
Č
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@@*
use_locking(

conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@*
T0
[
conv2d_1/ConstConst*
dtype0*
_output_shapes
:@*
valueB@*    
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
÷
conv2d_1/convolutionConv2Dleaky_re_lu_1/LeakyReluconv2d_1/kernel/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
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
:˙˙˙˙˙˙˙˙˙¸Đ@

leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_1/BiasAdd*
alpha%>*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0
s
dropout_1/IdentityIdentityleaky_re_lu_2/LeakyRelu*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
Ä
max_pooling2d_1/MaxPoolMaxPooldropout_1/Identity*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@
v
conv2d_2/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @      
`
conv2d_2/random_uniform/minConst*
valueB
 *ď[q˝*
dtype0*
_output_shapes
: 
`
conv2d_2/random_uniform/maxConst*
valueB
 *ď[q=*
dtype0*
_output_shapes
: 
ł
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*'
_output_shapes
:@*
seed2˝ŕ*
seedą˙ĺ)*
T0*
dtype0
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
T0*
_output_shapes
: 

conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*'
_output_shapes
:@*
T0

conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*
T0*'
_output_shapes
:@

conv2d_2/kernel
VariableV2*
dtype0*'
_output_shapes
:@*
	container *
shape:@*
shared_name 
É
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*'
_output_shapes
:@

conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*'
_output_shapes
:@
]
conv2d_2/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_2/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ž
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
u
conv2d_2/bias/readIdentityconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
_output_shapes	
:*
T0
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ř
conv2d_2/convolutionConv2Dmax_pooling2d_1/MaxPoolconv2d_2/kernel/read*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨

leaky_re_lu_3/LeakyRelu	LeakyReluconv2d_2/BiasAdd*
T0*
alpha%>*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
v
conv2d_3/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
`
conv2d_3/random_uniform/minConst*
valueB
 *ěQ˝*
dtype0*
_output_shapes
: 
`
conv2d_3/random_uniform/maxConst*
valueB
 *ěQ=*
dtype0*
_output_shapes
: 
´
%conv2d_3/random_uniform/RandomUniformRandomUniformconv2d_3/random_uniform/shape*
T0*
dtype0*(
_output_shapes
:*
seed2°ś*
seedą˙ĺ)
}
conv2d_3/random_uniform/subSubconv2d_3/random_uniform/maxconv2d_3/random_uniform/min*
_output_shapes
: *
T0

conv2d_3/random_uniform/mulMul%conv2d_3/random_uniform/RandomUniformconv2d_3/random_uniform/sub*(
_output_shapes
:*
T0

conv2d_3/random_uniformAddconv2d_3/random_uniform/mulconv2d_3/random_uniform/min*(
_output_shapes
:*
T0

conv2d_3/kernel
VariableV2*(
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
Ę
conv2d_3/kernel/AssignAssignconv2d_3/kernelconv2d_3/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(*(
_output_shapes
:

conv2d_3/kernel/readIdentityconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*(
_output_shapes
:*
T0
]
conv2d_3/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
{
conv2d_3/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ž
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/Const*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
u
conv2d_3/bias/readIdentityconv2d_3/bias*
T0* 
_class
loc:@conv2d_3/bias*
_output_shapes	
:
s
"conv2d_3/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ř
conv2d_3/convolutionConv2Dleaky_re_lu_3/LeakyReluconv2d_3/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨

conv2d_3/BiasAddBiasAddconv2d_3/convolutionconv2d_3/bias/read*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0

leaky_re_lu_4/LeakyRelu	LeakyReluconv2d_3/BiasAdd*
alpha%>*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0
t
dropout_2/IdentityIdentityleaky_re_lu_4/LeakyRelu*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
Ĺ
max_pooling2d_2/MaxPoolMaxPooldropout_2/Identity*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0
l
up_sampling2d_1/ShapeShapemax_pooling2d_2/MaxPool*
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
Í
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape#up_sampling2d_1/strided_slice/stack%up_sampling2d_1/strided_slice/stack_1%up_sampling2d_1/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0*
shrink_axis_mask 
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
ž
%up_sampling2d_1/ResizeNearestNeighborResizeNearestNeighbormax_pooling2d_2/MaxPoolup_sampling2d_1/mul*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
align_corners( 
v
conv2d_4/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   
`
conv2d_4/random_uniform/minConst*
valueB
 *ď[q˝*
dtype0*
_output_shapes
: 
`
conv2d_4/random_uniform/maxConst*
valueB
 *ď[q=*
dtype0*
_output_shapes
: 
ł
%conv2d_4/random_uniform/RandomUniformRandomUniformconv2d_4/random_uniform/shape*
T0*
dtype0*'
_output_shapes
:@*
seed2ÄÚ*
seedą˙ĺ)
}
conv2d_4/random_uniform/subSubconv2d_4/random_uniform/maxconv2d_4/random_uniform/min*
T0*
_output_shapes
: 

conv2d_4/random_uniform/mulMul%conv2d_4/random_uniform/RandomUniformconv2d_4/random_uniform/sub*'
_output_shapes
:@*
T0

conv2d_4/random_uniformAddconv2d_4/random_uniform/mulconv2d_4/random_uniform/min*
T0*'
_output_shapes
:@

conv2d_4/kernel
VariableV2*
dtype0*'
_output_shapes
:@*
	container *
shape:@*
shared_name 
É
conv2d_4/kernel/AssignAssignconv2d_4/kernelconv2d_4/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*'
_output_shapes
:@

conv2d_4/kernel/readIdentityconv2d_4/kernel*
T0*"
_class
loc:@conv2d_4/kernel*'
_output_shapes
:@
[
conv2d_4/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_4/bias
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
­
conv2d_4/bias/AssignAssignconv2d_4/biasconv2d_4/Const*
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
t
conv2d_4/bias/readIdentityconv2d_4/bias* 
_class
loc:@conv2d_4/bias*
_output_shapes
:@*
T0
s
"conv2d_4/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

conv2d_4/convolutionConv2D%up_sampling2d_1/ResizeNearestNeighborconv2d_4/kernel/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
	dilations
*
T0

conv2d_4/BiasAddBiasAddconv2d_4/convolutionconv2d_4/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@

leaky_re_lu_5/LeakyRelu	LeakyReluconv2d_4/BiasAdd*
T0*
alpha%>*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@
v
conv2d_5/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
`
conv2d_5/random_uniform/minConst*
valueB
 *:Í˝*
dtype0*
_output_shapes
: 
`
conv2d_5/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *:Í=
˛
%conv2d_5/random_uniform/RandomUniformRandomUniformconv2d_5/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*&
_output_shapes
:@@*
seed2ěĺ
}
conv2d_5/random_uniform/subSubconv2d_5/random_uniform/maxconv2d_5/random_uniform/min*
T0*
_output_shapes
: 

conv2d_5/random_uniform/mulMul%conv2d_5/random_uniform/RandomUniformconv2d_5/random_uniform/sub*&
_output_shapes
:@@*
T0

conv2d_5/random_uniformAddconv2d_5/random_uniform/mulconv2d_5/random_uniform/min*
T0*&
_output_shapes
:@@

conv2d_5/kernel
VariableV2*
shape:@@*
shared_name *
dtype0*&
_output_shapes
:@@*
	container 
Č
conv2d_5/kernel/AssignAssignconv2d_5/kernelconv2d_5/random_uniform*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel

conv2d_5/kernel/readIdentityconv2d_5/kernel*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:@@
[
conv2d_5/ConstConst*
dtype0*
_output_shapes
:@*
valueB@*    
y
conv2d_5/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
­
conv2d_5/bias/AssignAssignconv2d_5/biasconv2d_5/Const*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias*
validate_shape(
t
conv2d_5/bias/readIdentityconv2d_5/bias*
T0* 
_class
loc:@conv2d_5/bias*
_output_shapes
:@
s
"conv2d_5/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
÷
conv2d_5/convolutionConv2Dleaky_re_lu_5/LeakyReluconv2d_5/kernel/read*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv2d_5/BiasAddBiasAddconv2d_5/convolutionconv2d_5/bias/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
T0*
data_formatNHWC

leaky_re_lu_6/LeakyRelu	LeakyReluconv2d_5/BiasAdd*
T0*
alpha%>*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@
s
dropout_3/IdentityIdentityleaky_re_lu_6/LeakyRelu*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
T0
v
conv2d_6/random_uniform/shapeConst*
_output_shapes
:*%
valueB"      @      *
dtype0
`
conv2d_6/random_uniform/minConst*
valueB
 *Üž*
dtype0*
_output_shapes
: 
`
conv2d_6/random_uniform/maxConst*
valueB
 *Ü>*
dtype0*
_output_shapes
: 
ą
%conv2d_6/random_uniform/RandomUniformRandomUniformconv2d_6/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*&
_output_shapes
:@*
seed2ŐÖ
}
conv2d_6/random_uniform/subSubconv2d_6/random_uniform/maxconv2d_6/random_uniform/min*
_output_shapes
: *
T0

conv2d_6/random_uniform/mulMul%conv2d_6/random_uniform/RandomUniformconv2d_6/random_uniform/sub*
T0*&
_output_shapes
:@

conv2d_6/random_uniformAddconv2d_6/random_uniform/mulconv2d_6/random_uniform/min*
T0*&
_output_shapes
:@

conv2d_6/kernel
VariableV2*
dtype0*&
_output_shapes
:@*
	container *
shape:@*
shared_name 
Č
conv2d_6/kernel/AssignAssignconv2d_6/kernelconv2d_6/random_uniform*&
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(

conv2d_6/kernel/readIdentityconv2d_6/kernel*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:@*
T0
[
conv2d_6/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv2d_6/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
­
conv2d_6/bias/AssignAssignconv2d_6/biasconv2d_6/Const*
use_locking(*
T0* 
_class
loc:@conv2d_6/bias*
validate_shape(*
_output_shapes
:
t
conv2d_6/bias/readIdentityconv2d_6/bias*
_output_shapes
:*
T0* 
_class
loc:@conv2d_6/bias
s
"conv2d_6/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ň
conv2d_6/convolutionConv2Ddropout_3/Identityconv2d_6/kernel/read*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv2d_6/BiasAddBiasAddconv2d_6/convolutionconv2d_6/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨

output/DepthToSpaceDepthToSpaceconv2d_6/BiasAdd*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*

block_size*
T0*
data_formatNHWC
k
output/ResizeBilinear/sizeConst*
valueB"8  P  *
dtype0*
_output_shapes
:
Š
output/ResizeBilinearResizeBilinearoutput/DepthToSpaceoutput/ResizeBilinear/size*
align_corners( *
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ

output/PlaceholderPlaceholder*
dtype0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨*&
shape:˙˙˙˙˙˙˙˙˙¨

output/DepthToSpace_1DepthToSpaceoutput/Placeholder*

block_size*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
m
output/ResizeBilinear_1/sizeConst*
valueB"8  P  *
dtype0*
_output_shapes
:
Ż
output/ResizeBilinear_1ResizeBilinearoutput/DepthToSpace_1output/ResizeBilinear_1/size*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
align_corners( 
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
dtype0	*
_output_shapes
: *
	container *
shape: 
ž
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
 *ŹĹ'7*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
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
Adam/lr/readIdentityAdam/lr*
_class
loc:@Adam/lr*
_output_shapes
: *
T0
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
Ž
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
T0*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: *
use_locking(
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_class
loc:@Adam/beta_1*
_output_shapes
: 
^
Adam/beta_2/initial_valueConst*
valueB
 *wž?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
Ž
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_2*
validate_shape(*
_output_shapes
: 
j
Adam/beta_2/readIdentityAdam/beta_2*
T0*
_class
loc:@Adam/beta_2*
_output_shapes
: 
]
Adam/decay/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
n

Adam/decay
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
Ş
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
ś
output_targetPlaceholder*
dtype0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*?
shape6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p
output_sample_weightsPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
loss/output_loss/subSuboutput/ResizeBilinearoutput_target*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
m
loss/output_loss/AbsAbsloss/output_loss/sub*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
r
'loss/output_loss/Mean/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
ą
loss/output_loss/MeanMeanloss/output_loss/Abs'loss/output_loss/Mean/reduction_indices*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
	keep_dims( *

Tidx0
z
)loss/output_loss/Mean_1/reduction_indicesConst*
_output_shapes
:*
valueB"      *
dtype0
Ź
loss/output_loss/Mean_1Meanloss/output_loss/Mean)loss/output_loss/Mean_1/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *

Tidx0*
T0
y
loss/output_loss/mulMulloss/output_loss/Mean_1output_sample_weights*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
`
loss/output_loss/NotEqual/yConst*
dtype0*
_output_shapes
: *
valueB
 *    

loss/output_loss/NotEqualNotEqualoutput_sample_weightsloss/output_loss/NotEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

loss/output_loss/CastCastloss/output_loss/NotEqual*

SrcT0
*
Truncate( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
`
loss/output_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/output_loss/Mean_2Meanloss/output_loss/Castloss/output_loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

loss/output_loss/truedivRealDivloss/output_loss/mulloss/output_loss/Mean_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
loss/output_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/output_loss/Mean_3Meanloss/output_loss/truedivloss/output_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
ś
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*

index_type0*
_class
loc:@loss/mul*
_output_shapes
: 
Ľ
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/output_loss/Mean_3*
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
¸
Btraining/Adam/gradients/loss/output_loss/Mean_3_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:**
_class 
loc:@loss/output_loss/Mean_3

<training/Adam/gradients/loss/output_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Btraining/Adam/gradients/loss/output_loss/Mean_3_grad/Reshape/shape*
T0*
Tshape0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
:
ž
:training/Adam/gradients/loss/output_loss/Mean_3_grad/ShapeShapeloss/output_loss/truediv*
T0*
out_type0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
:
§
9training/Adam/gradients/loss/output_loss/Mean_3_grad/TileTile<training/Adam/gradients/loss/output_loss/Mean_3_grad/Reshape:training/Adam/gradients/loss/output_loss/Mean_3_grad/Shape**
_class 
loc:@loss/output_loss/Mean_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0
Ŕ
<training/Adam/gradients/loss/output_loss/Mean_3_grad/Shape_1Shapeloss/output_loss/truediv*
T0*
out_type0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
:
Ť
<training/Adam/gradients/loss/output_loss/Mean_3_grad/Shape_2Const*
valueB **
_class 
loc:@loss/output_loss/Mean_3*
dtype0*
_output_shapes
: 
°
:training/Adam/gradients/loss/output_loss/Mean_3_grad/ConstConst*
_output_shapes
:*
valueB: **
_class 
loc:@loss/output_loss/Mean_3*
dtype0
Ľ
9training/Adam/gradients/loss/output_loss/Mean_3_grad/ProdProd<training/Adam/gradients/loss/output_loss/Mean_3_grad/Shape_1:training/Adam/gradients/loss/output_loss/Mean_3_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/output_loss/Mean_3
˛
<training/Adam/gradients/loss/output_loss/Mean_3_grad/Const_1Const*
valueB: **
_class 
loc:@loss/output_loss/Mean_3*
dtype0*
_output_shapes
:
Š
;training/Adam/gradients/loss/output_loss/Mean_3_grad/Prod_1Prod<training/Adam/gradients/loss/output_loss/Mean_3_grad/Shape_2<training/Adam/gradients/loss/output_loss/Mean_3_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/output_loss/Mean_3
Ź
>training/Adam/gradients/loss/output_loss/Mean_3_grad/Maximum/yConst*
_output_shapes
: *
value	B :**
_class 
loc:@loss/output_loss/Mean_3*
dtype0

<training/Adam/gradients/loss/output_loss/Mean_3_grad/MaximumMaximum;training/Adam/gradients/loss/output_loss/Mean_3_grad/Prod_1>training/Adam/gradients/loss/output_loss/Mean_3_grad/Maximum/y*
T0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
: 

=training/Adam/gradients/loss/output_loss/Mean_3_grad/floordivFloorDiv9training/Adam/gradients/loss/output_loss/Mean_3_grad/Prod<training/Adam/gradients/loss/output_loss/Mean_3_grad/Maximum*
T0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
: 
ě
9training/Adam/gradients/loss/output_loss/Mean_3_grad/CastCast=training/Adam/gradients/loss/output_loss/Mean_3_grad/floordiv*

SrcT0**
_class 
loc:@loss/output_loss/Mean_3*
Truncate( *
_output_shapes
: *

DstT0

<training/Adam/gradients/loss/output_loss/Mean_3_grad/truedivRealDiv9training/Adam/gradients/loss/output_loss/Mean_3_grad/Tile9training/Adam/gradients/loss/output_loss/Mean_3_grad/Cast*
T0**
_class 
loc:@loss/output_loss/Mean_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
;training/Adam/gradients/loss/output_loss/truediv_grad/ShapeShapeloss/output_loss/mul*
T0*
out_type0*+
_class!
loc:@loss/output_loss/truediv*
_output_shapes
:
­
=training/Adam/gradients/loss/output_loss/truediv_grad/Shape_1Const*
valueB *+
_class!
loc:@loss/output_loss/truediv*
dtype0*
_output_shapes
: 
Ę
Ktraining/Adam/gradients/loss/output_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs;training/Adam/gradients/loss/output_loss/truediv_grad/Shape=training/Adam/gradients/loss/output_loss/truediv_grad/Shape_1*
T0*+
_class!
loc:@loss/output_loss/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ú
=training/Adam/gradients/loss/output_loss/truediv_grad/RealDivRealDiv<training/Adam/gradients/loss/output_loss/Mean_3_grad/truedivloss/output_loss/Mean_2*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
9training/Adam/gradients/loss/output_loss/truediv_grad/SumSum=training/Adam/gradients/loss/output_loss/truediv_grad/RealDivKtraining/Adam/gradients/loss/output_loss/truediv_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/output_loss/truediv*
_output_shapes
:
Š
=training/Adam/gradients/loss/output_loss/truediv_grad/ReshapeReshape9training/Adam/gradients/loss/output_loss/truediv_grad/Sum;training/Adam/gradients/loss/output_loss/truediv_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0*+
_class!
loc:@loss/output_loss/truediv
ą
9training/Adam/gradients/loss/output_loss/truediv_grad/NegNegloss/output_loss/mul*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
?training/Adam/gradients/loss/output_loss/truediv_grad/RealDiv_1RealDiv9training/Adam/gradients/loss/output_loss/truediv_grad/Negloss/output_loss/Mean_2*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
?training/Adam/gradients/loss/output_loss/truediv_grad/RealDiv_2RealDiv?training/Adam/gradients/loss/output_loss/truediv_grad/RealDiv_1loss/output_loss/Mean_2*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

9training/Adam/gradients/loss/output_loss/truediv_grad/mulMul<training/Adam/gradients/loss/output_loss/Mean_3_grad/truediv?training/Adam/gradients/loss/output_loss/truediv_grad/RealDiv_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*+
_class!
loc:@loss/output_loss/truediv
š
;training/Adam/gradients/loss/output_loss/truediv_grad/Sum_1Sum9training/Adam/gradients/loss/output_loss/truediv_grad/mulMtraining/Adam/gradients/loss/output_loss/truediv_grad/BroadcastGradientArgs:1*
T0*+
_class!
loc:@loss/output_loss/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
˘
?training/Adam/gradients/loss/output_loss/truediv_grad/Reshape_1Reshape;training/Adam/gradients/loss/output_loss/truediv_grad/Sum_1=training/Adam/gradients/loss/output_loss/truediv_grad/Shape_1*
Tshape0*+
_class!
loc:@loss/output_loss/truediv*
_output_shapes
: *
T0
ˇ
7training/Adam/gradients/loss/output_loss/mul_grad/ShapeShapeloss/output_loss/Mean_1*
T0*
out_type0*'
_class
loc:@loss/output_loss/mul*
_output_shapes
:
ˇ
9training/Adam/gradients/loss/output_loss/mul_grad/Shape_1Shapeoutput_sample_weights*
out_type0*'
_class
loc:@loss/output_loss/mul*
_output_shapes
:*
T0
ş
Gtraining/Adam/gradients/loss/output_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs7training/Adam/gradients/loss/output_loss/mul_grad/Shape9training/Adam/gradients/loss/output_loss/mul_grad/Shape_1*
T0*'
_class
loc:@loss/output_loss/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
é
5training/Adam/gradients/loss/output_loss/mul_grad/MulMul=training/Adam/gradients/loss/output_loss/truediv_grad/Reshapeoutput_sample_weights*
T0*'
_class
loc:@loss/output_loss/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
5training/Adam/gradients/loss/output_loss/mul_grad/SumSum5training/Adam/gradients/loss/output_loss/mul_grad/MulGtraining/Adam/gradients/loss/output_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*'
_class
loc:@loss/output_loss/mul

9training/Adam/gradients/loss/output_loss/mul_grad/ReshapeReshape5training/Adam/gradients/loss/output_loss/mul_grad/Sum7training/Adam/gradients/loss/output_loss/mul_grad/Shape*
T0*
Tshape0*'
_class
loc:@loss/output_loss/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
í
7training/Adam/gradients/loss/output_loss/mul_grad/Mul_1Mulloss/output_loss/Mean_1=training/Adam/gradients/loss/output_loss/truediv_grad/Reshape*'
_class
loc:@loss/output_loss/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ť
7training/Adam/gradients/loss/output_loss/mul_grad/Sum_1Sum7training/Adam/gradients/loss/output_loss/mul_grad/Mul_1Itraining/Adam/gradients/loss/output_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*'
_class
loc:@loss/output_loss/mul*
_output_shapes
:

;training/Adam/gradients/loss/output_loss/mul_grad/Reshape_1Reshape7training/Adam/gradients/loss/output_loss/mul_grad/Sum_19training/Adam/gradients/loss/output_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_class
loc:@loss/output_loss/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
:training/Adam/gradients/loss/output_loss/Mean_1_grad/ShapeShapeloss/output_loss/Mean*
out_type0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:*
T0
§
9training/Adam/gradients/loss/output_loss/Mean_1_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :**
_class 
loc:@loss/output_loss/Mean_1
ö
8training/Adam/gradients/loss/output_loss/Mean_1_grad/addAdd)loss/output_loss/Mean_1/reduction_indices9training/Adam/gradients/loss/output_loss/Mean_1_grad/Size*
_output_shapes
:*
T0**
_class 
loc:@loss/output_loss/Mean_1

8training/Adam/gradients/loss/output_loss/Mean_1_grad/modFloorMod8training/Adam/gradients/loss/output_loss/Mean_1_grad/add9training/Adam/gradients/loss/output_loss/Mean_1_grad/Size*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:
˛
<training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape_1Const*
valueB:**
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
:
Ž
@training/Adam/gradients/loss/output_loss/Mean_1_grad/range/startConst*
value	B : **
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
: 
Ž
@training/Adam/gradients/loss/output_loss/Mean_1_grad/range/deltaConst*
value	B :**
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
: 
Ö
:training/Adam/gradients/loss/output_loss/Mean_1_grad/rangeRange@training/Adam/gradients/loss/output_loss/Mean_1_grad/range/start9training/Adam/gradients/loss/output_loss/Mean_1_grad/Size@training/Adam/gradients/loss/output_loss/Mean_1_grad/range/delta*
_output_shapes
:*

Tidx0**
_class 
loc:@loss/output_loss/Mean_1
­
?training/Adam/gradients/loss/output_loss/Mean_1_grad/Fill/valueConst*
value	B :**
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
: 
Ł
9training/Adam/gradients/loss/output_loss/Mean_1_grad/FillFill<training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape_1?training/Adam/gradients/loss/output_loss/Mean_1_grad/Fill/value*
T0*

index_type0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:

Btraining/Adam/gradients/loss/output_loss/Mean_1_grad/DynamicStitchDynamicStitch:training/Adam/gradients/loss/output_loss/Mean_1_grad/range8training/Adam/gradients/loss/output_loss/Mean_1_grad/mod:training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape9training/Adam/gradients/loss/output_loss/Mean_1_grad/Fill*
_output_shapes
:*
T0**
_class 
loc:@loss/output_loss/Mean_1*
N
Ź
>training/Adam/gradients/loss/output_loss/Mean_1_grad/Maximum/yConst*
value	B :**
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
: 

<training/Adam/gradients/loss/output_loss/Mean_1_grad/MaximumMaximumBtraining/Adam/gradients/loss/output_loss/Mean_1_grad/DynamicStitch>training/Adam/gradients/loss/output_loss/Mean_1_grad/Maximum/y*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:

=training/Adam/gradients/loss/output_loss/Mean_1_grad/floordivFloorDiv:training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape<training/Adam/gradients/loss/output_loss/Mean_1_grad/Maximum*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:
Č
<training/Adam/gradients/loss/output_loss/Mean_1_grad/ReshapeReshape9training/Adam/gradients/loss/output_loss/mul_grad/ReshapeBtraining/Adam/gradients/loss/output_loss/Mean_1_grad/DynamicStitch*
T0*
Tshape0**
_class 
loc:@loss/output_loss/Mean_1*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ä
9training/Adam/gradients/loss/output_loss/Mean_1_grad/TileTile<training/Adam/gradients/loss/output_loss/Mean_1_grad/Reshape=training/Adam/gradients/loss/output_loss/Mean_1_grad/floordiv*

Tmultiples0*
T0**
_class 
loc:@loss/output_loss/Mean_1*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
˝
<training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape_2Shapeloss/output_loss/Mean*
_output_shapes
:*
T0*
out_type0**
_class 
loc:@loss/output_loss/Mean_1
ż
<training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape_3Shapeloss/output_loss/Mean_1*
T0*
out_type0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:
°
:training/Adam/gradients/loss/output_loss/Mean_1_grad/ConstConst*
valueB: **
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
:
Ľ
9training/Adam/gradients/loss/output_loss/Mean_1_grad/ProdProd<training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape_2:training/Adam/gradients/loss/output_loss/Mean_1_grad/Const*

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: 
˛
<training/Adam/gradients/loss/output_loss/Mean_1_grad/Const_1Const*
valueB: **
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
:
Š
;training/Adam/gradients/loss/output_loss/Mean_1_grad/Prod_1Prod<training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape_3<training/Adam/gradients/loss/output_loss/Mean_1_grad/Const_1*

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: 
Ž
@training/Adam/gradients/loss/output_loss/Mean_1_grad/Maximum_1/yConst*
value	B :**
_class 
loc:@loss/output_loss/Mean_1*
dtype0*
_output_shapes
: 

>training/Adam/gradients/loss/output_loss/Mean_1_grad/Maximum_1Maximum;training/Adam/gradients/loss/output_loss/Mean_1_grad/Prod_1@training/Adam/gradients/loss/output_loss/Mean_1_grad/Maximum_1/y*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: 

?training/Adam/gradients/loss/output_loss/Mean_1_grad/floordiv_1FloorDiv9training/Adam/gradients/loss/output_loss/Mean_1_grad/Prod>training/Adam/gradients/loss/output_loss/Mean_1_grad/Maximum_1*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: 
î
9training/Adam/gradients/loss/output_loss/Mean_1_grad/CastCast?training/Adam/gradients/loss/output_loss/Mean_1_grad/floordiv_1*

SrcT0**
_class 
loc:@loss/output_loss/Mean_1*
Truncate( *
_output_shapes
: *

DstT0
Ą
<training/Adam/gradients/loss/output_loss/Mean_1_grad/truedivRealDiv9training/Adam/gradients/loss/output_loss/Mean_1_grad/Tile9training/Adam/gradients/loss/output_loss/Mean_1_grad/Cast*
T0**
_class 
loc:@loss/output_loss/Mean_1*-
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
ś
8training/Adam/gradients/loss/output_loss/Mean_grad/ShapeShapeloss/output_loss/Abs*
T0*
out_type0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:
Ł
7training/Adam/gradients/loss/output_loss/Mean_grad/SizeConst*
value	B :*(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 
ę
6training/Adam/gradients/loss/output_loss/Mean_grad/addAdd'loss/output_loss/Mean/reduction_indices7training/Adam/gradients/loss/output_loss/Mean_grad/Size*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: *
T0
ţ
6training/Adam/gradients/loss/output_loss/Mean_grad/modFloorMod6training/Adam/gradients/loss/output_loss/Mean_grad/add7training/Adam/gradients/loss/output_loss/Mean_grad/Size*
_output_shapes
: *
T0*(
_class
loc:@loss/output_loss/Mean
§
:training/Adam/gradients/loss/output_loss/Mean_grad/Shape_1Const*
valueB *(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 
Ş
>training/Adam/gradients/loss/output_loss/Mean_grad/range/startConst*
value	B : *(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 
Ş
>training/Adam/gradients/loss/output_loss/Mean_grad/range/deltaConst*
value	B :*(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 
Ě
8training/Adam/gradients/loss/output_loss/Mean_grad/rangeRange>training/Adam/gradients/loss/output_loss/Mean_grad/range/start7training/Adam/gradients/loss/output_loss/Mean_grad/Size>training/Adam/gradients/loss/output_loss/Mean_grad/range/delta*
_output_shapes
:*

Tidx0*(
_class
loc:@loss/output_loss/Mean
Š
=training/Adam/gradients/loss/output_loss/Mean_grad/Fill/valueConst*
value	B :*(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
: 

7training/Adam/gradients/loss/output_loss/Mean_grad/FillFill:training/Adam/gradients/loss/output_loss/Mean_grad/Shape_1=training/Adam/gradients/loss/output_loss/Mean_grad/Fill/value*
T0*

index_type0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 

@training/Adam/gradients/loss/output_loss/Mean_grad/DynamicStitchDynamicStitch8training/Adam/gradients/loss/output_loss/Mean_grad/range6training/Adam/gradients/loss/output_loss/Mean_grad/mod8training/Adam/gradients/loss/output_loss/Mean_grad/Shape7training/Adam/gradients/loss/output_loss/Mean_grad/Fill*
T0*(
_class
loc:@loss/output_loss/Mean*
N*
_output_shapes
:
¨
<training/Adam/gradients/loss/output_loss/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*(
_class
loc:@loss/output_loss/Mean*
dtype0

:training/Adam/gradients/loss/output_loss/Mean_grad/MaximumMaximum@training/Adam/gradients/loss/output_loss/Mean_grad/DynamicStitch<training/Adam/gradients/loss/output_loss/Mean_grad/Maximum/y*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:

;training/Adam/gradients/loss/output_loss/Mean_grad/floordivFloorDiv8training/Adam/gradients/loss/output_loss/Mean_grad/Shape:training/Adam/gradients/loss/output_loss/Mean_grad/Maximum*
_output_shapes
:*
T0*(
_class
loc:@loss/output_loss/Mean
Ň
:training/Adam/gradients/loss/output_loss/Mean_grad/ReshapeReshape<training/Adam/gradients/loss/output_loss/Mean_1_grad/truediv@training/Adam/gradients/loss/output_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*(
_class
loc:@loss/output_loss/Mean*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
É
7training/Adam/gradients/loss/output_loss/Mean_grad/TileTile:training/Adam/gradients/loss/output_loss/Mean_grad/Reshape;training/Adam/gradients/loss/output_loss/Mean_grad/floordiv*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0*(
_class
loc:@loss/output_loss/Mean
¸
:training/Adam/gradients/loss/output_loss/Mean_grad/Shape_2Shapeloss/output_loss/Abs*
T0*
out_type0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:
š
:training/Adam/gradients/loss/output_loss/Mean_grad/Shape_3Shapeloss/output_loss/Mean*
T0*
out_type0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:
Ź
8training/Adam/gradients/loss/output_loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *(
_class
loc:@loss/output_loss/Mean

7training/Adam/gradients/loss/output_loss/Mean_grad/ProdProd:training/Adam/gradients/loss/output_loss/Mean_grad/Shape_28training/Adam/gradients/loss/output_loss/Mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/output_loss/Mean
Ž
:training/Adam/gradients/loss/output_loss/Mean_grad/Const_1Const*
valueB: *(
_class
loc:@loss/output_loss/Mean*
dtype0*
_output_shapes
:
Ą
9training/Adam/gradients/loss/output_loss/Mean_grad/Prod_1Prod:training/Adam/gradients/loss/output_loss/Mean_grad/Shape_3:training/Adam/gradients/loss/output_loss/Mean_grad/Const_1*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ş
>training/Adam/gradients/loss/output_loss/Mean_grad/Maximum_1/yConst*
_output_shapes
: *
value	B :*(
_class
loc:@loss/output_loss/Mean*
dtype0

<training/Adam/gradients/loss/output_loss/Mean_grad/Maximum_1Maximum9training/Adam/gradients/loss/output_loss/Mean_grad/Prod_1>training/Adam/gradients/loss/output_loss/Mean_grad/Maximum_1/y*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 

=training/Adam/gradients/loss/output_loss/Mean_grad/floordiv_1FloorDiv7training/Adam/gradients/loss/output_loss/Mean_grad/Prod<training/Adam/gradients/loss/output_loss/Mean_grad/Maximum_1*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
č
7training/Adam/gradients/loss/output_loss/Mean_grad/CastCast=training/Adam/gradients/loss/output_loss/Mean_grad/floordiv_1*(
_class
loc:@loss/output_loss/Mean*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0

:training/Adam/gradients/loss/output_loss/Mean_grad/truedivRealDiv7training/Adam/gradients/loss/output_loss/Mean_grad/Tile7training/Adam/gradients/loss/output_loss/Mean_grad/Cast*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
T0*(
_class
loc:@loss/output_loss/Mean
š
6training/Adam/gradients/loss/output_loss/Abs_grad/SignSignloss/output_loss/sub*'
_class
loc:@loss/output_loss/Abs*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
T0

5training/Adam/gradients/loss/output_loss/Abs_grad/mulMul:training/Adam/gradients/loss/output_loss/Mean_grad/truediv6training/Adam/gradients/loss/output_loss/Abs_grad/Sign*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
T0*'
_class
loc:@loss/output_loss/Abs
ľ
7training/Adam/gradients/loss/output_loss/sub_grad/ShapeShapeoutput/ResizeBilinear*
_output_shapes
:*
T0*
out_type0*'
_class
loc:@loss/output_loss/sub
Ż
9training/Adam/gradients/loss/output_loss/sub_grad/Shape_1Shapeoutput_target*
out_type0*'
_class
loc:@loss/output_loss/sub*
_output_shapes
:*
T0
ş
Gtraining/Adam/gradients/loss/output_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs7training/Adam/gradients/loss/output_loss/sub_grad/Shape9training/Adam/gradients/loss/output_loss/sub_grad/Shape_1*
T0*'
_class
loc:@loss/output_loss/sub*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ľ
5training/Adam/gradients/loss/output_loss/sub_grad/SumSum5training/Adam/gradients/loss/output_loss/Abs_grad/mulGtraining/Adam/gradients/loss/output_loss/sub_grad/BroadcastGradientArgs*
T0*'
_class
loc:@loss/output_loss/sub*
_output_shapes
:*

Tidx0*
	keep_dims( 
§
9training/Adam/gradients/loss/output_loss/sub_grad/ReshapeReshape5training/Adam/gradients/loss/output_loss/sub_grad/Sum7training/Adam/gradients/loss/output_loss/sub_grad/Shape*
T0*
Tshape0*'
_class
loc:@loss/output_loss/sub*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
Š
7training/Adam/gradients/loss/output_loss/sub_grad/Sum_1Sum5training/Adam/gradients/loss/output_loss/Abs_grad/mulItraining/Adam/gradients/loss/output_loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*'
_class
loc:@loss/output_loss/sub
Á
5training/Adam/gradients/loss/output_loss/sub_grad/NegNeg7training/Adam/gradients/loss/output_loss/sub_grad/Sum_1*
_output_shapes
:*
T0*'
_class
loc:@loss/output_loss/sub
Ä
;training/Adam/gradients/loss/output_loss/sub_grad/Reshape_1Reshape5training/Adam/gradients/loss/output_loss/sub_grad/Neg9training/Adam/gradients/loss/output_loss/sub_grad/Shape_1*
T0*
Tshape0*'
_class
loc:@loss/output_loss/sub*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ś
Etraining/Adam/gradients/output/ResizeBilinear_grad/ResizeBilinearGradResizeBilinearGrad9training/Adam/gradients/loss/output_loss/sub_grad/Reshapeoutput/DepthToSpace*(
_class
loc:@output/ResizeBilinear*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
align_corners( *
T0
Ą
=training/Adam/gradients/output/DepthToSpace_grad/SpaceToDepthSpaceToDepthEtraining/Adam/gradients/output/ResizeBilinear_grad/ResizeBilinearGrad*
T0*&
_class
loc:@output/DepthToSpace*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨*

block_size
č
9training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGradBiasAddGrad=training/Adam/gradients/output/DepthToSpace_grad/SpaceToDepth*#
_class
loc:@conv2d_6/BiasAdd*
data_formatNHWC*
_output_shapes
:*
T0
Ů
8training/Adam/gradients/conv2d_6/convolution_grad/ShapeNShapeNdropout_3/Identityconv2d_6/kernel/read*
out_type0*'
_class
loc:@conv2d_6/convolution*
N* 
_output_shapes
::*
T0
ž
Etraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_6/convolution_grad/ShapeNconv2d_6/kernel/read=training/Adam/gradients/output/DepthToSpace_grad/SpaceToDepth*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
	dilations
*
T0*'
_class
loc:@conv2d_6/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ľ
Ftraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdropout_3/Identity:training/Adam/gradients/conv2d_6/convolution_grad/ShapeN:1=training/Adam/gradients/output/DepthToSpace_grad/SpaceToDepth*
paddingSAME*&
_output_shapes
:@*
	dilations
*
T0*'
_class
loc:@conv2d_6/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
¤
Btraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGradLeakyReluGradEtraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputconv2d_5/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*
alpha%>*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@
í
9training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_5/BiasAdd*
data_formatNHWC*
_output_shapes
:@
Ţ
8training/Adam/gradients/conv2d_5/convolution_grad/ShapeNShapeNleaky_re_lu_5/LeakyReluconv2d_5/kernel/read* 
_output_shapes
::*
T0*
out_type0*'
_class
loc:@conv2d_5/convolution*
N
Ă
Etraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_5/convolution_grad/ShapeNconv2d_5/kernel/readBtraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGrad*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
	dilations
*
T0*'
_class
loc:@conv2d_5/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
ż
Ftraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterleaky_re_lu_5/LeakyRelu:training/Adam/gradients/conv2d_5/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_5/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@@
¤
Btraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGradLeakyReluGradEtraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropInputconv2d_4/BiasAdd*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
T0**
_class 
loc:@leaky_re_lu_5/LeakyRelu*
alpha%>
í
9training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_4/BiasAdd*
data_formatNHWC*
_output_shapes
:@
ě
8training/Adam/gradients/conv2d_4/convolution_grad/ShapeNShapeN%up_sampling2d_1/ResizeNearestNeighborconv2d_4/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_4/convolution*
N* 
_output_shapes
::
Ä
Etraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_4/convolution_grad/ShapeNconv2d_4/kernel/readBtraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
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
Î
Ftraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_1/ResizeNearestNeighbor:training/Adam/gradients/conv2d_4/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*'
_class
loc:@conv2d_4/convolution
ě
atraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
dtype0*
_output_shapes
:*
valueB"    *8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor
Ż
\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradEtraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputatraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
align_corners( *
T0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙

@training/Adam/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGraddropout_2/Identitymax_pooling2d_2/MaxPool\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0**
_class 
loc:@max_pooling2d_2/MaxPool*
data_formatNHWC*
strides
*
ksize

 
Btraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGradLeakyReluGrad@training/Adam/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradconv2d_3/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*
alpha%>*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
î
9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGrad*
_output_shapes	
:*
T0*#
_class
loc:@conv2d_3/BiasAdd*
data_formatNHWC
Ţ
8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNShapeNleaky_re_lu_3/LeakyReluconv2d_3/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_3/convolution*
N* 
_output_shapes
::
Ä
Etraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNconv2d_3/kernel/readBtraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
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
Á
Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterleaky_re_lu_3/LeakyRelu:training/Adam/gradients/conv2d_3/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGrad*(
_output_shapes
:*
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
Ľ
Btraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGradLeakyReluGradEtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputconv2d_2/BiasAdd*
alpha%>*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0**
_class 
loc:@leaky_re_lu_3/LeakyRelu
î
9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_2/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ţ
8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNShapeNmax_pooling2d_1/MaxPoolconv2d_2/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_2/convolution*
N* 
_output_shapes
::
Ă
Etraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNconv2d_2/kernel/readBtraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGrad*
T0*'
_class
loc:@conv2d_2/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
	dilations

Ŕ
Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_1/MaxPool:training/Adam/gradients/conv2d_2/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGrad*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
ý
@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGraddropout_1/Identitymax_pooling2d_1/MaxPoolEtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0**
_class 
loc:@max_pooling2d_1/MaxPool*
data_formatNHWC*
strides
*
ksize
*
paddingSAME

Btraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGradLeakyReluGrad@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradconv2d_1/BiasAdd*
alpha%>*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu
í
9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_1/BiasAdd*
data_formatNHWC*
_output_shapes
:@
Ţ
8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNShapeNleaky_re_lu_1/LeakyReluconv2d_1/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_1/convolution*
N* 
_output_shapes
::
Ă
Etraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNconv2d_1/kernel/readBtraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
ż
Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterleaky_re_lu_1/LeakyRelu:training/Adam/gradients/conv2d_1/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
T0*'
_class
loc:@conv2d_1/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@@*
	dilations

Ą
Btraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGradLeakyReluGradEtraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputinput/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
alpha%>*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
ç
6training/Adam/gradients/input/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
T0* 
_class
loc:@input/BiasAdd*
data_formatNHWC*
_output_shapes
:@
É
5training/Adam/gradients/input/convolution_grad/ShapeNShapeNinput_inputinput/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0*$
_class
loc:@input/convolution
ˇ
Btraining/Adam/gradients/input/convolution_grad/Conv2DBackpropInputConv2DBackpropInput5training/Adam/gradients/input/convolution_grad/ShapeNinput/kernel/readBtraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
	dilations
*
T0*$
_class
loc:@input/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
Ş
Ctraining/Adam/gradients/input/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterinput_input7training/Adam/gradients/input/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*$
_class
loc:@input/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@*
	dilations
*
T0
_
training/Adam/AssignAdd/valueConst*
_output_shapes
: *
value	B	 R*
dtype0	
Ź
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
training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
_output_shapes
: *
T0
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
_output_shapes
: *
T0
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
training/Adam/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
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
Ů
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
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
Ő
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
¤
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*&
_output_shapes
:@@*
T0*

index_type0

training/Adam/Variable_2
VariableV2*
shape:@@*
shared_name *
dtype0*&
_output_shapes
:@@*
	container 
á
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*&
_output_shapes
:@@
Ą
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*&
_output_shapes
:@@*
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
VariableV2*
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
Ő
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
Ľ
training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_4
VariableV2*
shape:@*
shared_name *
dtype0*'
_output_shapes
:@*
	container 
â
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*'
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(
˘
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*'
_output_shapes
:@*
T0*+
_class!
loc:@training/Adam/Variable_4
d
training/Adam/zeros_5Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_5
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ö
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes	
:
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
training/Adam/zeros_6/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ś
training/Adam/zeros_6Fill%training/Adam/zeros_6/shape_as_tensortraining/Adam/zeros_6/Const*
T0*

index_type0*(
_output_shapes
:
 
training/Adam/Variable_6
VariableV2*
shared_name *
dtype0*(
_output_shapes
:*
	container *
shape:
ă
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*(
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(
Ł
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*(
_output_shapes
:*
T0*+
_class!
loc:@training/Adam/Variable_6
d
training/Adam/zeros_7Const*
_output_shapes	
:*
valueB*    *
dtype0

training/Adam/Variable_7
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Ö
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes	
:
~
%training/Adam/zeros_8/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_8/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ľ
training/Adam/zeros_8Fill%training/Adam/zeros_8/shape_as_tensortraining/Adam/zeros_8/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_8
VariableV2*
shared_name *
dtype0*'
_output_shapes
:@*
	container *
shape:@
â
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*'
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(
˘
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
T0*+
_class!
loc:@training/Adam/Variable_8*'
_output_shapes
:@
b
training/Adam/zeros_9Const*
dtype0*
_output_shapes
:@*
valueB@*    

training/Adam/Variable_9
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
Ő
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:@*
use_locking(

training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
_output_shapes
:@*
T0*+
_class!
loc:@training/Adam/Variable_9
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
§
training/Adam/zeros_10Fill&training/Adam/zeros_10/shape_as_tensortraining/Adam/zeros_10/Const*
T0*

index_type0*&
_output_shapes
:@@

training/Adam/Variable_10
VariableV2*
dtype0*&
_output_shapes
:@@*
	container *
shape:@@*
shared_name 
ĺ
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0
¤
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
Ů
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:@*
use_locking(
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
ĺ
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*&
_output_shapes
:@*
use_locking(
¤
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
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
Ů
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:

training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
:*
T0

&training/Adam/zeros_14/shape_as_tensorConst*
_output_shapes
:*%
valueB"         @   *
dtype0
a
training/Adam/zeros_14/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
§
training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*&
_output_shapes
:@*
T0*

index_type0

training/Adam/Variable_14
VariableV2*&
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
ĺ
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*&
_output_shapes
:@
¤
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*&
_output_shapes
:@
c
training/Adam/zeros_15Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_15
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
Ů
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
:@

&training/Adam/zeros_16/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*

index_type0*&
_output_shapes
:@@*
T0

training/Adam/Variable_16
VariableV2*&
_output_shapes
:@@*
	container *
shape:@@*
shared_name *
dtype0
ĺ
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*&
_output_shapes
:@@
¤
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
Ů
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
¨
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_18
VariableV2*
shape:@*
shared_name *
dtype0*'
_output_shapes
:@*
	container 
ć
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*'
_output_shapes
:@
Ľ
training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*'
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_18
e
training/Adam/zeros_19Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_19
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ú
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
_output_shapes	
:

&training/Adam/zeros_20/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"            
a
training/Adam/zeros_20/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_20
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ç
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(
Ś
training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
T0*,
_class"
 loc:@training/Adam/Variable_20*(
_output_shapes
:
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
Ú
 training/Adam/Variable_21/AssignAssigntraining/Adam/Variable_21training/Adam/zeros_21*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*
T0*,
_class"
 loc:@training/Adam/Variable_21*
_output_shapes	
:
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
¨
training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_22
VariableV2*
shape:@*
shared_name *
dtype0*'
_output_shapes
:@*
	container 
ć
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22
Ľ
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
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
Ů
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0

training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes
:@*
T0

&training/Adam/zeros_24/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
a
training/Adam/zeros_24/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
training/Adam/zeros_24Fill&training/Adam/zeros_24/shape_as_tensortraining/Adam/zeros_24/Const*
T0*

index_type0*&
_output_shapes
:@@

training/Adam/Variable_24
VariableV2*
shape:@@*
shared_name *
dtype0*&
_output_shapes
:@@*
	container 
ĺ
 training/Adam/Variable_24/AssignAssigntraining/Adam/Variable_24training/Adam/zeros_24*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(*&
_output_shapes
:@@
¤
training/Adam/Variable_24/readIdentitytraining/Adam/Variable_24*&
_output_shapes
:@@*
T0*,
_class"
 loc:@training/Adam/Variable_24
c
training/Adam/zeros_25Const*
_output_shapes
:@*
valueB@*    *
dtype0

training/Adam/Variable_25
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
Ů
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
training/Adam/zeros_26Const*
dtype0*&
_output_shapes
:@*%
valueB@*    

training/Adam/Variable_26
VariableV2*&
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
ĺ
 training/Adam/Variable_26/AssignAssigntraining/Adam/Variable_26training/Adam/zeros_26*
T0*,
_class"
 loc:@training/Adam/Variable_26*
validate_shape(*&
_output_shapes
:@*
use_locking(
¤
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
Ů
 training/Adam/Variable_27/AssignAssigntraining/Adam/Variable_27training/Adam/zeros_27*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_27*
validate_shape(*
_output_shapes
:

training/Adam/Variable_27/readIdentitytraining/Adam/Variable_27*,
_class"
 loc:@training/Adam/Variable_27*
_output_shapes
:*
T0
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
Ů
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
training/Adam/zeros_29/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_29Fill&training/Adam/zeros_29/shape_as_tensortraining/Adam/zeros_29/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_29
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
Ů
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
&training/Adam/zeros_30/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
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
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ů
 training/Adam/Variable_30/AssignAssigntraining/Adam/Variable_30training/Adam/zeros_30*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_30*
validate_shape(*
_output_shapes
:

training/Adam/Variable_30/readIdentitytraining/Adam/Variable_30*
T0*,
_class"
 loc:@training/Adam/Variable_30*
_output_shapes
:
p
&training/Adam/zeros_31/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_31/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_31Fill&training/Adam/zeros_31/shape_as_tensortraining/Adam/zeros_31/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_31
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ů
 training/Adam/Variable_31/AssignAssigntraining/Adam/Variable_31training/Adam/zeros_31*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_31

training/Adam/Variable_31/readIdentitytraining/Adam/Variable_31*
T0*,
_class"
 loc:@training/Adam/Variable_31*
_output_shapes
:
p
&training/Adam/zeros_32/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
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
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ů
 training/Adam/Variable_32/AssignAssigntraining/Adam/Variable_32training/Adam/zeros_32*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_32*
validate_shape(

training/Adam/Variable_32/readIdentitytraining/Adam/Variable_32*
T0*,
_class"
 loc:@training/Adam/Variable_32*
_output_shapes
:
p
&training/Adam/zeros_33/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_33/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_33Fill&training/Adam/zeros_33/shape_as_tensortraining/Adam/zeros_33/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_33
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ů
 training/Adam/Variable_33/AssignAssigntraining/Adam/Variable_33training/Adam/zeros_33*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_33*
validate_shape(
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
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
Ů
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
Ů
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
training/Adam/zeros_36/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_36Fill&training/Adam/zeros_36/shape_as_tensortraining/Adam/zeros_36/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_36
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ů
 training/Adam/Variable_36/AssignAssigntraining/Adam/Variable_36training/Adam/zeros_36*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_36*
validate_shape(*
_output_shapes
:

training/Adam/Variable_36/readIdentitytraining/Adam/Variable_36*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_36
p
&training/Adam/zeros_37/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
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
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ů
 training/Adam/Variable_37/AssignAssigntraining/Adam/Variable_37training/Adam/zeros_37*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_37*
validate_shape(*
_output_shapes
:
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
training/Adam/zeros_38/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_38Fill&training/Adam/zeros_38/shape_as_tensortraining/Adam/zeros_38/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_38
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
Ů
 training/Adam/Variable_38/AssignAssigntraining/Adam/Variable_38training/Adam/zeros_38*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_38*
validate_shape(*
_output_shapes
:

training/Adam/Variable_38/readIdentitytraining/Adam/Variable_38*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_38
p
&training/Adam/zeros_39/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_39/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_39Fill&training/Adam/zeros_39/shape_as_tensortraining/Adam/zeros_39/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_39
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ů
 training/Adam/Variable_39/AssignAssigntraining/Adam/Variable_39training/Adam/zeros_39*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_39
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
training/Adam/zeros_40Fill&training/Adam/zeros_40/shape_as_tensortraining/Adam/zeros_40/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_40
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ů
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
training/Adam/zeros_41Fill&training/Adam/zeros_41/shape_as_tensortraining/Adam/zeros_41/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_41
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ů
 training/Adam/Variable_41/AssignAssigntraining/Adam/Variable_41training/Adam/zeros_41*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_41*
validate_shape(*
_output_shapes
:
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
Ľ
training/Adam/mul_2Multraining/Adam/sub_2Ctraining/Adam/gradients/input/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@*
T0
u
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*&
_output_shapes
:@*
T0
}
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_14/read*&
_output_shapes
:@*
T0
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

training/Adam/SquareSquareCtraining/Adam/gradients/input/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@*
T0
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
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*&
_output_shapes
:@
Z
training/Adam/add_3/yConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
x
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*&
_output_shapes
:@
}
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*&
_output_shapes
:@*
T0
w
training/Adam/sub_4Subinput/kernel/readtraining/Adam/truediv_1*&
_output_shapes
:@*
T0
Đ
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*)
_class
loc:@training/Adam/Variable*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
Ř
training/Adam/Assign_1Assigntraining/Adam/Variable_14training/Adam/add_2*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*&
_output_shapes
:@
ž
training/Adam/Assign_2Assigninput/kerneltraining/Adam/sub_4*
T0*
_class
loc:@input/kernel*
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
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_7Multraining/Adam/sub_56training/Adam/gradients/input/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
:@
q
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_15/read*
_output_shapes
:@*
T0
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
}
training/Adam/Square_1Square6training/Adam/gradients/input/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes
:@
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
_output_shapes
:@*
T0
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
_output_shapes
:@*
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
:@

training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
_output_shapes
:@*
T0
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
_output_shapes
:@*
T0
Z
training/Adam/add_6/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes
:@
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes
:@
i
training/Adam/sub_7Subinput/bias/readtraining/Adam/truediv_2*
_output_shapes
:@*
T0
Ę
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:@
Ě
training/Adam/Assign_4Assigntraining/Adam/Variable_15training/Adam/add_5*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:@
Ž
training/Adam/Assign_5Assign
input/biastraining/Adam/sub_7*
use_locking(*
T0*
_class
loc:@input/bias*
validate_shape(*
_output_shapes
:@
}
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*&
_output_shapes
:@@*
T0
Z
training/Adam/sub_8/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
_output_shapes
: *
T0
Š
training/Adam/mul_12Multraining/Adam/sub_8Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
w
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*&
_output_shapes
:@@*
T0
~
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_16/read*&
_output_shapes
:@@*
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
training/Adam/Square_2SquareFtraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@@
y
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*&
_output_shapes
:@@
w
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*&
_output_shapes
:@@
t
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*&
_output_shapes
:@@
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
:@@

training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
T0*&
_output_shapes
:@@
l
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*&
_output_shapes
:@@
Z
training/Adam/add_9/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
x
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0*&
_output_shapes
:@@
~
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*&
_output_shapes
:@@
{
training/Adam/sub_10Subconv2d_1/kernel/readtraining/Adam/truediv_3*
T0*&
_output_shapes
:@@
Ö
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*&
_output_shapes
:@@
Ř
training/Adam/Assign_7Assigntraining/Adam/Variable_16training/Adam/add_8*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*&
_output_shapes
:@@
Ĺ
training/Adam/Assign_8Assignconv2d_1/kerneltraining/Adam/sub_10*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@@
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
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_17Multraining/Adam/sub_119training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
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
training/Adam/Square_3Square9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
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
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
:@
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
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
_output_shapes
:@*
T0
[
training/Adam/add_12/yConst*
valueB
 *żÖ3*
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
training/Adam/sub_13Subconv2d_1/bias/readtraining/Adam/truediv_4*
T0*
_output_shapes
:@
Ë
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:@
Î
training/Adam/Assign_10Assigntraining/Adam/Variable_17training/Adam/add_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:@
ś
training/Adam/Assign_11Assignconv2d_1/biastraining/Adam/sub_13*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
~
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0*'
_output_shapes
:@
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
Ť
training/Adam/mul_22Multraining/Adam/sub_14Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
y
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*'
_output_shapes
:@*
T0
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
training/Adam/Square_4SquareFtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
{
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*'
_output_shapes
:@
y
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*'
_output_shapes
:@*
T0
v
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*'
_output_shapes
:@
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
training/Adam/add_15/yConst*
_output_shapes
: *
valueB
 *żÖ3*
dtype0
{
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*'
_output_shapes
:@

training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*'
_output_shapes
:@
|
training/Adam/sub_16Subconv2d_2/kernel/readtraining/Adam/truediv_5*
T0*'
_output_shapes
:@
Ů
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*'
_output_shapes
:@*
use_locking(
Ű
training/Adam/Assign_13Assigntraining/Adam/Variable_18training/Adam/add_14*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0
Ç
training/Adam/Assign_14Assignconv2d_2/kerneltraining/Adam/sub_16*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*'
_output_shapes
:@*
use_locking(
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
training/Adam/mul_27Multraining/Adam/sub_179training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
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
training/Adam/Square_5Square9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
_output_shapes	
:*
T0
m
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes	
:
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
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
_output_shapes	
:*
T0
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
 *żÖ3*
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
training/Adam/sub_19Subconv2d_2/bias/readtraining/Adam/truediv_6*
_output_shapes	
:*
T0
Í
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_16Assigntraining/Adam/Variable_19training/Adam/add_17*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes	
:*
use_locking(
ˇ
training/Adam/Assign_17Assignconv2d_2/biastraining/Adam/sub_19*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias

training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*(
_output_shapes
:*
T0
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
Ź
training/Adam/mul_32Multraining/Adam/sub_20Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
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
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_6SquareFtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
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
training/Adam/Const_14Const*
dtype0*
_output_shapes
: *
valueB
 *    
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
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
T0*(
_output_shapes
:
n
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0*(
_output_shapes
:
[
training/Adam/add_21/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
|
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
T0*(
_output_shapes
:

training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*(
_output_shapes
:
}
training/Adam/sub_22Subconv2d_3/kernel/readtraining/Adam/truediv_7*
T0*(
_output_shapes
:
Ú
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_19Assigntraining/Adam/Variable_20training/Adam/add_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(*(
_output_shapes
:
Č
training/Adam/Assign_20Assignconv2d_3/kerneltraining/Adam/sub_22*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(*(
_output_shapes
:
r
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
T0*
_output_shapes	
:
[
training/Adam/sub_23/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_37Multraining/Adam/sub_239training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
_output_shapes	
:*
T0
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
training/Adam/Square_7Square9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
_output_shapes	
:*
T0
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
training/Adam/Const_16Const*
_output_shapes
: *
valueB
 *    *
dtype0
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
training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_16*
T0*
_output_shapes	
:
a
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
_output_shapes	
:*
T0
[
training/Adam/add_24/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
o
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
T0*
_output_shapes	
:
t
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
T0*
_output_shapes	
:
n
training/Adam/sub_25Subconv2d_3/bias/readtraining/Adam/truediv_8*
T0*
_output_shapes	
:
Í
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ď
training/Adam/Assign_22Assigntraining/Adam/Variable_21training/Adam/add_23*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes	
:
ˇ
training/Adam/Assign_23Assignconv2d_3/biastraining/Adam/sub_25*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes	
:
~
training/Adam/mul_41MulAdam/beta_1/readtraining/Adam/Variable_8/read*'
_output_shapes
:@*
T0
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
Ť
training/Adam/mul_42Multraining/Adam/sub_26Ftraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
y
training/Adam/add_25Addtraining/Adam/mul_41training/Adam/mul_42*
T0*'
_output_shapes
:@

training/Adam/mul_43MulAdam/beta_2/readtraining/Adam/Variable_22/read*'
_output_shapes
:@*
T0
[
training/Adam/sub_27/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_27Subtraining/Adam/sub_27/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_8SquareFtraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
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
training/Adam/mul_45Multraining/Adam/multraining/Adam/add_25*'
_output_shapes
:@*
T0
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
training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_18*'
_output_shapes
:@*
T0
m
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
T0*'
_output_shapes
:@
[
training/Adam/add_27/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
{
training/Adam/add_27Addtraining/Adam/Sqrt_9training/Adam/add_27/y*'
_output_shapes
:@*
T0

training/Adam/truediv_9RealDivtraining/Adam/mul_45training/Adam/add_27*'
_output_shapes
:@*
T0
|
training/Adam/sub_28Subconv2d_4/kernel/readtraining/Adam/truediv_9*
T0*'
_output_shapes
:@
Ů
training/Adam/Assign_24Assigntraining/Adam/Variable_8training/Adam/add_25*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0
Ű
training/Adam/Assign_25Assigntraining/Adam/Variable_22training/Adam/add_26*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0
Ç
training/Adam/Assign_26Assignconv2d_4/kerneltraining/Adam/sub_28*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel
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
training/Adam/mul_47Multraining/Adam/sub_299training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
l
training/Adam/add_28Addtraining/Adam/mul_46training/Adam/mul_47*
_output_shapes
:@*
T0
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
training/Adam/sub_30Subtraining/Adam/sub_30/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_9Square9training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
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
training/Adam/mul_50Multraining/Adam/multraining/Adam/add_28*
_output_shapes
:@*
T0
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
 *żÖ3*
dtype0*
_output_shapes
: 
o
training/Adam/add_30Addtraining/Adam/Sqrt_10training/Adam/add_30/y*
_output_shapes
:@*
T0
t
training/Adam/truediv_10RealDivtraining/Adam/mul_50training/Adam/add_30*
T0*
_output_shapes
:@
n
training/Adam/sub_31Subconv2d_4/bias/readtraining/Adam/truediv_10*
_output_shapes
:@*
T0
Ě
training/Adam/Assign_27Assigntraining/Adam/Variable_9training/Adam/add_28*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:@
Î
training/Adam/Assign_28Assigntraining/Adam/Variable_23training/Adam/add_29*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes
:@
ś
training/Adam/Assign_29Assignconv2d_4/biastraining/Adam/sub_31*
use_locking(*
T0* 
_class
loc:@conv2d_4/bias*
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
Ş
training/Adam/mul_52Multraining/Adam/sub_32Ftraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
x
training/Adam/add_31Addtraining/Adam/mul_51training/Adam/mul_52*
T0*&
_output_shapes
:@@
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
training/Adam/Square_10SquareFtraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*&
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
training/Adam/Const_22Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_23Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_11/MinimumMinimumtraining/Adam/add_32training/Adam/Const_23*&
_output_shapes
:@@*
T0

training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_22*
T0*&
_output_shapes
:@@
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
 *żÖ3*
dtype0
{
training/Adam/add_33Addtraining/Adam/Sqrt_11training/Adam/add_33/y*
T0*&
_output_shapes
:@@

training/Adam/truediv_11RealDivtraining/Adam/mul_55training/Adam/add_33*
T0*&
_output_shapes
:@@
|
training/Adam/sub_34Subconv2d_5/kernel/readtraining/Adam/truediv_11*
T0*&
_output_shapes
:@@
Ú
training/Adam/Assign_30Assigntraining/Adam/Variable_10training/Adam/add_31*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10
Ú
training/Adam/Assign_31Assigntraining/Adam/Variable_24training/Adam/add_32*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(*&
_output_shapes
:@@
Ć
training/Adam/Assign_32Assignconv2d_5/kerneltraining/Adam/sub_34*
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel*
validate_shape(*&
_output_shapes
:@@
r
training/Adam/mul_56MulAdam/beta_1/readtraining/Adam/Variable_11/read*
_output_shapes
:@*
T0
[
training/Adam/sub_35/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_35Subtraining/Adam/sub_35/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_57Multraining/Adam/sub_359training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
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
training/Adam/Square_11Square9training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
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
training/Adam/Const_25Const*
valueB
 *  *
dtype0*
_output_shapes
: 
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
training/Adam/Sqrt_12Sqrttraining/Adam/clip_by_value_12*
_output_shapes
:@*
T0
[
training/Adam/add_36/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
o
training/Adam/add_36Addtraining/Adam/Sqrt_12training/Adam/add_36/y*
_output_shapes
:@*
T0
t
training/Adam/truediv_12RealDivtraining/Adam/mul_60training/Adam/add_36*
_output_shapes
:@*
T0
n
training/Adam/sub_37Subconv2d_5/bias/readtraining/Adam/truediv_12*
_output_shapes
:@*
T0
Î
training/Adam/Assign_33Assigntraining/Adam/Variable_11training/Adam/add_34*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:@*
use_locking(
Î
training/Adam/Assign_34Assigntraining/Adam/Variable_25training/Adam/add_35*
T0*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(*
_output_shapes
:@*
use_locking(
ś
training/Adam/Assign_35Assignconv2d_5/biastraining/Adam/sub_37*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias*
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
training/Adam/sub_38Subtraining/Adam/sub_38/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ş
training/Adam/mul_62Multraining/Adam/sub_38Ftraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@*
T0
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
training/Adam/sub_39/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_39Subtraining/Adam/sub_39/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_12SquareFtraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@*
T0
{
training/Adam/mul_64Multraining/Adam/sub_39training/Adam/Square_12*&
_output_shapes
:@*
T0
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
 *żÖ3*
dtype0*
_output_shapes
: 
{
training/Adam/add_39Addtraining/Adam/Sqrt_13training/Adam/add_39/y*
T0*&
_output_shapes
:@

training/Adam/truediv_13RealDivtraining/Adam/mul_65training/Adam/add_39*
T0*&
_output_shapes
:@
|
training/Adam/sub_40Subconv2d_6/kernel/readtraining/Adam/truediv_13*
T0*&
_output_shapes
:@
Ú
training/Adam/Assign_36Assigntraining/Adam/Variable_12training/Adam/add_37*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*&
_output_shapes
:@
Ú
training/Adam/Assign_37Assigntraining/Adam/Variable_26training/Adam/add_38*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_26
Ć
training/Adam/Assign_38Assignconv2d_6/kerneltraining/Adam/sub_40*"
_class
loc:@conv2d_6/kernel*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
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
training/Adam/sub_41Subtraining/Adam/sub_41/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_67Multraining/Adam/sub_419training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
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
training/Adam/Square_13Square9training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
o
training/Adam/mul_69Multraining/Adam/sub_42training/Adam/Square_13*
T0*
_output_shapes
:
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
training/Adam/clip_by_value_14Maximum&training/Adam/clip_by_value_14/Minimumtraining/Adam/Const_28*
_output_shapes
:*
T0
b
training/Adam/Sqrt_14Sqrttraining/Adam/clip_by_value_14*
_output_shapes
:*
T0
[
training/Adam/add_42/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
o
training/Adam/add_42Addtraining/Adam/Sqrt_14training/Adam/add_42/y*
T0*
_output_shapes
:
t
training/Adam/truediv_14RealDivtraining/Adam/mul_70training/Adam/add_42*
T0*
_output_shapes
:
n
training/Adam/sub_43Subconv2d_6/bias/readtraining/Adam/truediv_14*
T0*
_output_shapes
:
Î
training/Adam/Assign_39Assigntraining/Adam/Variable_13training/Adam/add_40*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:
Î
training/Adam/Assign_40Assigntraining/Adam/Variable_27training/Adam/add_41*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_27
ś
training/Adam/Assign_41Assignconv2d_6/biastraining/Adam/sub_43*
use_locking(*
T0* 
_class
loc:@conv2d_6/bias*
validate_shape(*
_output_shapes
:
ř
training/group_depsNoOp	^loss/mul^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_24^training/Adam/Assign_25^training/Adam/Assign_26^training/Adam/Assign_27^training/Adam/Assign_28^training/Adam/Assign_29^training/Adam/Assign_3^training/Adam/Assign_30^training/Adam/Assign_31^training/Adam/Assign_32^training/Adam/Assign_33^training/Adam/Assign_34^training/Adam/Assign_35^training/Adam/Assign_36^training/Adam/Assign_37^training/Adam/Assign_38^training/Adam/Assign_39^training/Adam/Assign_4^training/Adam/Assign_40^training/Adam/Assign_41^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9


group_depsNoOp	^loss/mul

IsVariableInitializedIsVariableInitializedinput/kernel*
dtype0*
_output_shapes
: *
_class
loc:@input/kernel
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
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializedconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializedconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_6IsVariableInitializedconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_7IsVariableInitializedconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializedconv2d_4/kernel*
_output_shapes
: *"
_class
loc:@conv2d_4/kernel*
dtype0

IsVariableInitialized_9IsVariableInitializedconv2d_4/bias*
dtype0*
_output_shapes
: * 
_class
loc:@conv2d_4/bias

IsVariableInitialized_10IsVariableInitializedconv2d_5/kernel*"
_class
loc:@conv2d_5/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_11IsVariableInitializedconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_12IsVariableInitializedconv2d_6/kernel*"
_class
loc:@conv2d_6/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_13IsVariableInitializedconv2d_6/bias*
_output_shapes
: * 
_class
loc:@conv2d_6/bias*
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
IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable*
_output_shapes
: *)
_class
loc:@training/Adam/Variable*
dtype0

IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_2*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_2
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
IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes
: 

IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_7*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_7
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
IsVariableInitialized_34IsVariableInitializedtraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
dtype0*
_output_shapes
: 

IsVariableInitialized_35IsVariableInitializedtraining/Adam/Variable_16*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_16
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
IsVariableInitialized_39IsVariableInitializedtraining/Adam/Variable_20*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_20

IsVariableInitialized_40IsVariableInitializedtraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
: 

IsVariableInitialized_41IsVariableInitializedtraining/Adam/Variable_22*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_22*
dtype0

IsVariableInitialized_42IsVariableInitializedtraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
dtype0*
_output_shapes
: 

IsVariableInitialized_43IsVariableInitializedtraining/Adam/Variable_24*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_24

IsVariableInitialized_44IsVariableInitializedtraining/Adam/Variable_25*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_25
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
IsVariableInitialized_47IsVariableInitializedtraining/Adam/Variable_28*,
_class"
 loc:@training/Adam/Variable_28*
dtype0*
_output_shapes
: 
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
IsVariableInitialized_53IsVariableInitializedtraining/Adam/Variable_34*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_34*
dtype0

IsVariableInitialized_54IsVariableInitializedtraining/Adam/Variable_35*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_35
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
IsVariableInitialized_58IsVariableInitializedtraining/Adam/Variable_39*,
_class"
 loc:@training/Adam/Variable_39*
dtype0*
_output_shapes
: 
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
đ
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^conv2d_3/bias/Assign^conv2d_3/kernel/Assign^conv2d_4/bias/Assign^conv2d_4/kernel/Assign^conv2d_5/bias/Assign^conv2d_5/kernel/Assign^conv2d_6/bias/Assign^conv2d_6/kernel/Assign^input/bias/Assign^input/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign!^training/Adam/Variable_24/Assign!^training/Adam/Variable_25/Assign!^training/Adam/Variable_26/Assign!^training/Adam/Variable_27/Assign!^training/Adam/Variable_28/Assign!^training/Adam/Variable_29/Assign ^training/Adam/Variable_3/Assign!^training/Adam/Variable_30/Assign!^training/Adam/Variable_31/Assign!^training/Adam/Variable_32/Assign!^training/Adam/Variable_33/Assign!^training/Adam/Variable_34/Assign!^training/Adam/Variable_35/Assign!^training/Adam/Variable_36/Assign!^training/Adam/Variable_37/Assign!^training/Adam/Variable_38/Assign!^training/Adam/Variable_39/Assign ^training/Adam/Variable_4/Assign!^training/Adam/Variable_40/Assign!^training/Adam/Variable_41/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign"S°EpîŤ     ¤bţH	Ie4H5×AJá×

Ů+ˇ+
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
ě
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
alphafloat%ÍĚL>"
Ttype0:
2
n
LeakyReluGrad
	gradients"T
features"T
	backprops"T"
alphafloat%ÍĚL>"
Ttype0:
2
Ô
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
î
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
ö
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
shared_namestring *1.13.12v1.13.1-0-g6612da8951ĺž	

input_inputPlaceholder*
dtype0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*&
shape:˙˙˙˙˙˙˙˙˙¸Đ
s
input/random_uniform/shapeConst*
_output_shapes
:*%
valueB"         @   *
dtype0
]
input/random_uniform/minConst*
valueB
 *8JĚ˝*
dtype0*
_output_shapes
: 
]
input/random_uniform/maxConst*
valueB
 *8JĚ=*
dtype0*
_output_shapes
: 
Ź
"input/random_uniform/RandomUniformRandomUniforminput/random_uniform/shape*
seed2÷Ú*&
_output_shapes
:@*
seedą˙ĺ)*
T0*
dtype0
t
input/random_uniform/subSubinput/random_uniform/maxinput/random_uniform/min*
T0*
_output_shapes
: 

input/random_uniform/mulMul"input/random_uniform/RandomUniforminput/random_uniform/sub*
T0*&
_output_shapes
:@

input/random_uniformAddinput/random_uniform/mulinput/random_uniform/min*
T0*&
_output_shapes
:@

input/kernel
VariableV2*
	container *&
_output_shapes
:@*
shape:@*
shared_name *
dtype0
ź
input/kernel/AssignAssigninput/kernelinput/random_uniform*
use_locking(*
T0*
_class
loc:@input/kernel*
validate_shape(*&
_output_shapes
:@
}
input/kernel/readIdentityinput/kernel*
T0*
_class
loc:@input/kernel*&
_output_shapes
:@
X
input/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
v

input/bias
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
Ą
input/bias/AssignAssign
input/biasinput/Const*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@input/bias
k
input/bias/readIdentity
input/bias*
T0*
_class
loc:@input/bias*
_output_shapes
:@
p
input/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ĺ
input/convolutionConv2Dinput_inputinput/kernel/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME

input/BiasAddBiasAddinput/convolutioninput/bias/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0*
data_formatNHWC

leaky_re_lu_1/LeakyRelu	LeakyReluinput/BiasAdd*
alpha%>*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0
v
conv2d_1/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
`
conv2d_1/random_uniform/minConst*
valueB
 *:Í˝*
dtype0*
_output_shapes
: 
`
conv2d_1/random_uniform/maxConst*
valueB
 *:Í=*
dtype0*
_output_shapes
: 
˛
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*
seed2ó¤*&
_output_shapes
:@@
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
T0*
_output_shapes
: 

conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*
T0*&
_output_shapes
:@@

conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*
T0*&
_output_shapes
:@@

conv2d_1/kernel
VariableV2*
shape:@@*
shared_name *
dtype0*
	container *&
_output_shapes
:@@
Č
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@@

conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@*
T0
[
conv2d_1/ConstConst*
dtype0*
_output_shapes
:@*
valueB@*    
y
conv2d_1/bias
VariableV2*
shape:@*
shared_name *
dtype0*
	container *
_output_shapes
:@
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
conv2d_1/bias/readIdentityconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0
s
"conv2d_1/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
÷
conv2d_1/convolutionConv2Dleaky_re_lu_1/LeakyReluconv2d_1/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
	dilations
*
T0

conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@

leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_1/BiasAdd*
T0*
alpha%>*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
s
dropout_1/IdentityIdentityleaky_re_lu_2/LeakyRelu*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
Ä
max_pooling2d_1/MaxPoolMaxPooldropout_1/Identity*
ksize
*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
T0*
data_formatNHWC*
strides

v
conv2d_2/random_uniform/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
`
conv2d_2/random_uniform/minConst*
valueB
 *ď[q˝*
dtype0*
_output_shapes
: 
`
conv2d_2/random_uniform/maxConst*
valueB
 *ď[q=*
dtype0*
_output_shapes
: 
ł
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
seed2˝ŕ*'
_output_shapes
:@*
seedą˙ĺ)*
T0*
dtype0
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
_output_shapes
: *
T0

conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*
T0*'
_output_shapes
:@

conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*'
_output_shapes
:@*
T0

conv2d_2/kernel
VariableV2*
dtype0*
	container *'
_output_shapes
:@*
shape:@*
shared_name 
É
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*'
_output_shapes
:@

conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*'
_output_shapes
:@
]
conv2d_2/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_2/bias
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
Ž
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes	
:
u
conv2d_2/bias/readIdentityconv2d_2/bias*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes	
:
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ř
conv2d_2/convolutionConv2Dmax_pooling2d_1/MaxPoolconv2d_2/kernel/read*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨

leaky_re_lu_3/LeakyRelu	LeakyReluconv2d_2/BiasAdd*
T0*
alpha%>*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
v
conv2d_3/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
`
conv2d_3/random_uniform/minConst*
valueB
 *ěQ˝*
dtype0*
_output_shapes
: 
`
conv2d_3/random_uniform/maxConst*
valueB
 *ěQ=*
dtype0*
_output_shapes
: 
´
%conv2d_3/random_uniform/RandomUniformRandomUniformconv2d_3/random_uniform/shape*
seed2°ś*(
_output_shapes
:*
seedą˙ĺ)*
T0*
dtype0
}
conv2d_3/random_uniform/subSubconv2d_3/random_uniform/maxconv2d_3/random_uniform/min*
T0*
_output_shapes
: 

conv2d_3/random_uniform/mulMul%conv2d_3/random_uniform/RandomUniformconv2d_3/random_uniform/sub*
T0*(
_output_shapes
:

conv2d_3/random_uniformAddconv2d_3/random_uniform/mulconv2d_3/random_uniform/min*
T0*(
_output_shapes
:

conv2d_3/kernel
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
Ę
conv2d_3/kernel/AssignAssignconv2d_3/kernelconv2d_3/random_uniform*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(

conv2d_3/kernel/readIdentityconv2d_3/kernel*
T0*"
_class
loc:@conv2d_3/kernel*(
_output_shapes
:
]
conv2d_3/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_3/bias
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ž
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/Const* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
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
ř
conv2d_3/convolutionConv2Dleaky_re_lu_3/LeakyReluconv2d_3/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨

conv2d_3/BiasAddBiasAddconv2d_3/convolutionconv2d_3/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨

leaky_re_lu_4/LeakyRelu	LeakyReluconv2d_3/BiasAdd*
T0*
alpha%>*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
t
dropout_2/IdentityIdentityleaky_re_lu_4/LeakyRelu*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
Ĺ
max_pooling2d_2/MaxPoolMaxPooldropout_2/Identity*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
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
Í
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape#up_sampling2d_1/strided_slice/stack%up_sampling2d_1/strided_slice/stack_1%up_sampling2d_1/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
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
ž
%up_sampling2d_1/ResizeNearestNeighborResizeNearestNeighbormax_pooling2d_2/MaxPoolup_sampling2d_1/mul*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
align_corners( 
v
conv2d_4/random_uniform/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
`
conv2d_4/random_uniform/minConst*
valueB
 *ď[q˝*
dtype0*
_output_shapes
: 
`
conv2d_4/random_uniform/maxConst*
valueB
 *ď[q=*
dtype0*
_output_shapes
: 
ł
%conv2d_4/random_uniform/RandomUniformRandomUniformconv2d_4/random_uniform/shape*
T0*
dtype0*
seed2ÄÚ*'
_output_shapes
:@*
seedą˙ĺ)
}
conv2d_4/random_uniform/subSubconv2d_4/random_uniform/maxconv2d_4/random_uniform/min*
_output_shapes
: *
T0

conv2d_4/random_uniform/mulMul%conv2d_4/random_uniform/RandomUniformconv2d_4/random_uniform/sub*
T0*'
_output_shapes
:@

conv2d_4/random_uniformAddconv2d_4/random_uniform/mulconv2d_4/random_uniform/min*'
_output_shapes
:@*
T0

conv2d_4/kernel
VariableV2*
shape:@*
shared_name *
dtype0*
	container *'
_output_shapes
:@
É
conv2d_4/kernel/AssignAssignconv2d_4/kernelconv2d_4/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*'
_output_shapes
:@

conv2d_4/kernel/readIdentityconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*'
_output_shapes
:@*
T0
[
conv2d_4/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_4/bias
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
­
conv2d_4/bias/AssignAssignconv2d_4/biasconv2d_4/Const*
use_locking(*
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(*
_output_shapes
:@
t
conv2d_4/bias/readIdentityconv2d_4/bias*
T0* 
_class
loc:@conv2d_4/bias*
_output_shapes
:@
s
"conv2d_4/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

conv2d_4/convolutionConv2D%up_sampling2d_1/ResizeNearestNeighborconv2d_4/kernel/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME

conv2d_4/BiasAddBiasAddconv2d_4/convolutionconv2d_4/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@

leaky_re_lu_5/LeakyRelu	LeakyReluconv2d_4/BiasAdd*
alpha%>*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
T0
v
conv2d_5/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
`
conv2d_5/random_uniform/minConst*
valueB
 *:Í˝*
dtype0*
_output_shapes
: 
`
conv2d_5/random_uniform/maxConst*
valueB
 *:Í=*
dtype0*
_output_shapes
: 
˛
%conv2d_5/random_uniform/RandomUniformRandomUniformconv2d_5/random_uniform/shape*
T0*
dtype0*
seed2ěĺ*&
_output_shapes
:@@*
seedą˙ĺ)
}
conv2d_5/random_uniform/subSubconv2d_5/random_uniform/maxconv2d_5/random_uniform/min*
T0*
_output_shapes
: 

conv2d_5/random_uniform/mulMul%conv2d_5/random_uniform/RandomUniformconv2d_5/random_uniform/sub*&
_output_shapes
:@@*
T0

conv2d_5/random_uniformAddconv2d_5/random_uniform/mulconv2d_5/random_uniform/min*
T0*&
_output_shapes
:@@

conv2d_5/kernel
VariableV2*
shared_name *
dtype0*
	container *&
_output_shapes
:@@*
shape:@@
Č
conv2d_5/kernel/AssignAssignconv2d_5/kernelconv2d_5/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel*
validate_shape(*&
_output_shapes
:@@

conv2d_5/kernel/readIdentityconv2d_5/kernel*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:@@
[
conv2d_5/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_5/bias
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
­
conv2d_5/bias/AssignAssignconv2d_5/biasconv2d_5/Const*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias*
validate_shape(
t
conv2d_5/bias/readIdentityconv2d_5/bias*
T0* 
_class
loc:@conv2d_5/bias*
_output_shapes
:@
s
"conv2d_5/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
÷
conv2d_5/convolutionConv2Dleaky_re_lu_5/LeakyReluconv2d_5/kernel/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

conv2d_5/BiasAddBiasAddconv2d_5/convolutionconv2d_5/bias/read*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
T0

leaky_re_lu_6/LeakyRelu	LeakyReluconv2d_5/BiasAdd*
T0*
alpha%>*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@
s
dropout_3/IdentityIdentityleaky_re_lu_6/LeakyRelu*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@
v
conv2d_6/random_uniform/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
`
conv2d_6/random_uniform/minConst*
valueB
 *Üž*
dtype0*
_output_shapes
: 
`
conv2d_6/random_uniform/maxConst*
valueB
 *Ü>*
dtype0*
_output_shapes
: 
ą
%conv2d_6/random_uniform/RandomUniformRandomUniformconv2d_6/random_uniform/shape*
T0*
dtype0*
seed2ŐÖ*&
_output_shapes
:@*
seedą˙ĺ)
}
conv2d_6/random_uniform/subSubconv2d_6/random_uniform/maxconv2d_6/random_uniform/min*
T0*
_output_shapes
: 

conv2d_6/random_uniform/mulMul%conv2d_6/random_uniform/RandomUniformconv2d_6/random_uniform/sub*
T0*&
_output_shapes
:@

conv2d_6/random_uniformAddconv2d_6/random_uniform/mulconv2d_6/random_uniform/min*
T0*&
_output_shapes
:@

conv2d_6/kernel
VariableV2*
	container *&
_output_shapes
:@*
shape:@*
shared_name *
dtype0
Č
conv2d_6/kernel/AssignAssignconv2d_6/kernelconv2d_6/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*&
_output_shapes
:@

conv2d_6/kernel/readIdentityconv2d_6/kernel*
T0*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:@
[
conv2d_6/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv2d_6/bias
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
­
conv2d_6/bias/AssignAssignconv2d_6/biasconv2d_6/Const*
use_locking(*
T0* 
_class
loc:@conv2d_6/bias*
validate_shape(*
_output_shapes
:
t
conv2d_6/bias/readIdentityconv2d_6/bias*
T0* 
_class
loc:@conv2d_6/bias*
_output_shapes
:
s
"conv2d_6/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ň
conv2d_6/convolutionConv2Ddropout_3/Identityconv2d_6/kernel/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME

conv2d_6/BiasAddBiasAddconv2d_6/convolutionconv2d_6/bias/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨*
T0*
data_formatNHWC

output/DepthToSpaceDepthToSpaceconv2d_6/BiasAdd*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*

block_size
k
output/ResizeBilinear/sizeConst*
_output_shapes
:*
valueB"8  P  *
dtype0
Š
output/ResizeBilinearResizeBilinearoutput/DepthToSpaceoutput/ResizeBilinear/size*
align_corners( *
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ

output/PlaceholderPlaceholder*
dtype0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨*&
shape:˙˙˙˙˙˙˙˙˙¨

output/DepthToSpace_1DepthToSpaceoutput/Placeholder*

block_size*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
m
output/ResizeBilinear_1/sizeConst*
valueB"8  P  *
dtype0*
_output_shapes
:
Ż
output/ResizeBilinear_1ResizeBilinearoutput/DepthToSpace_1output/ResizeBilinear_1/size*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
align_corners( 
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
ž
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
Adam/lr/initial_valueConst*
valueB
 *ŹĹ'7*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
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
Adam/beta_1/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
Adam/beta_1
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
shape: 
Ž
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
 *wž?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
shape: 
Ž
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_2*
validate_shape(*
_output_shapes
: 
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
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 
Ş
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
Adam/decay*
_output_shapes
: *
T0*
_class
loc:@Adam/decay
ś
output_targetPlaceholder*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*?
shape6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0
p
output_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
}
loss/output_loss/subSuboutput/ResizeBilinearoutput_target*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
m
loss/output_loss/AbsAbsloss/output_loss/sub*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
r
'loss/output_loss/Mean/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
ą
loss/output_loss/MeanMeanloss/output_loss/Abs'loss/output_loss/Mean/reduction_indices*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*

Tidx0*
	keep_dims( 
z
)loss/output_loss/Mean_1/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
Ź
loss/output_loss/Mean_1Meanloss/output_loss/Mean)loss/output_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
	keep_dims( 
y
loss/output_loss/mulMulloss/output_loss/Mean_1output_sample_weights*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
:˙˙˙˙˙˙˙˙˙

loss/output_loss/CastCastloss/output_loss/NotEqual*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
loss/output_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/output_loss/Mean_2Meanloss/output_loss/Castloss/output_loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0

loss/output_loss/truedivRealDivloss/output_loss/mulloss/output_loss/Mean_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
loss/output_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/output_loss/Mean_3Meanloss/output_loss/truedivloss/output_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
O

loss/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
U
loss/mulMul
loss/mul/xloss/output_loss/Mean_3*
T0*
_output_shapes
: 
}
training/Adam/gradients/ShapeConst*
_output_shapes
: *
_class
loc:@loss/mul*
valueB *
dtype0

!training/Adam/gradients/grad_ys_0Const*
_output_shapes
: *
_class
loc:@loss/mul*
valueB
 *  ?*
dtype0
ś
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
_output_shapes
: *
T0*
_class
loc:@loss/mul*

index_type0
Ľ
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/output_loss/Mean_3*
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
¸
Btraining/Adam/gradients/loss/output_loss/Mean_3_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:**
_class 
loc:@loss/output_loss/Mean_3*
valueB:

<training/Adam/gradients/loss/output_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Btraining/Adam/gradients/loss/output_loss/Mean_3_grad/Reshape/shape**
_class 
loc:@loss/output_loss/Mean_3*
Tshape0*
_output_shapes
:*
T0
ž
:training/Adam/gradients/loss/output_loss/Mean_3_grad/ShapeShapeloss/output_loss/truediv*
T0**
_class 
loc:@loss/output_loss/Mean_3*
out_type0*
_output_shapes
:
§
9training/Adam/gradients/loss/output_loss/Mean_3_grad/TileTile<training/Adam/gradients/loss/output_loss/Mean_3_grad/Reshape:training/Adam/gradients/loss/output_loss/Mean_3_grad/Shape**
_class 
loc:@loss/output_loss/Mean_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0
Ŕ
<training/Adam/gradients/loss/output_loss/Mean_3_grad/Shape_1Shapeloss/output_loss/truediv*
_output_shapes
:*
T0**
_class 
loc:@loss/output_loss/Mean_3*
out_type0
Ť
<training/Adam/gradients/loss/output_loss/Mean_3_grad/Shape_2Const*
_output_shapes
: **
_class 
loc:@loss/output_loss/Mean_3*
valueB *
dtype0
°
:training/Adam/gradients/loss/output_loss/Mean_3_grad/ConstConst**
_class 
loc:@loss/output_loss/Mean_3*
valueB: *
dtype0*
_output_shapes
:
Ľ
9training/Adam/gradients/loss/output_loss/Mean_3_grad/ProdProd<training/Adam/gradients/loss/output_loss/Mean_3_grad/Shape_1:training/Adam/gradients/loss/output_loss/Mean_3_grad/Const**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
˛
<training/Adam/gradients/loss/output_loss/Mean_3_grad/Const_1Const*
_output_shapes
:**
_class 
loc:@loss/output_loss/Mean_3*
valueB: *
dtype0
Š
;training/Adam/gradients/loss/output_loss/Mean_3_grad/Prod_1Prod<training/Adam/gradients/loss/output_loss/Mean_3_grad/Shape_2<training/Adam/gradients/loss/output_loss/Mean_3_grad/Const_1*
T0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
: *
	keep_dims( *

Tidx0
Ź
>training/Adam/gradients/loss/output_loss/Mean_3_grad/Maximum/yConst*
_output_shapes
: **
_class 
loc:@loss/output_loss/Mean_3*
value	B :*
dtype0

<training/Adam/gradients/loss/output_loss/Mean_3_grad/MaximumMaximum;training/Adam/gradients/loss/output_loss/Mean_3_grad/Prod_1>training/Adam/gradients/loss/output_loss/Mean_3_grad/Maximum/y*
T0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
: 

=training/Adam/gradients/loss/output_loss/Mean_3_grad/floordivFloorDiv9training/Adam/gradients/loss/output_loss/Mean_3_grad/Prod<training/Adam/gradients/loss/output_loss/Mean_3_grad/Maximum*
T0**
_class 
loc:@loss/output_loss/Mean_3*
_output_shapes
: 
ě
9training/Adam/gradients/loss/output_loss/Mean_3_grad/CastCast=training/Adam/gradients/loss/output_loss/Mean_3_grad/floordiv*

SrcT0**
_class 
loc:@loss/output_loss/Mean_3*
Truncate( *

DstT0*
_output_shapes
: 

<training/Adam/gradients/loss/output_loss/Mean_3_grad/truedivRealDiv9training/Adam/gradients/loss/output_loss/Mean_3_grad/Tile9training/Adam/gradients/loss/output_loss/Mean_3_grad/Cast*
T0**
_class 
loc:@loss/output_loss/Mean_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
;training/Adam/gradients/loss/output_loss/truediv_grad/ShapeShapeloss/output_loss/mul*
T0*+
_class!
loc:@loss/output_loss/truediv*
out_type0*
_output_shapes
:
­
=training/Adam/gradients/loss/output_loss/truediv_grad/Shape_1Const*+
_class!
loc:@loss/output_loss/truediv*
valueB *
dtype0*
_output_shapes
: 
Ę
Ktraining/Adam/gradients/loss/output_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs;training/Adam/gradients/loss/output_loss/truediv_grad/Shape=training/Adam/gradients/loss/output_loss/truediv_grad/Shape_1*
T0*+
_class!
loc:@loss/output_loss/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ú
=training/Adam/gradients/loss/output_loss/truediv_grad/RealDivRealDiv<training/Adam/gradients/loss/output_loss/Mean_3_grad/truedivloss/output_loss/Mean_2*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
9training/Adam/gradients/loss/output_loss/truediv_grad/SumSum=training/Adam/gradients/loss/output_loss/truediv_grad/RealDivKtraining/Adam/gradients/loss/output_loss/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/output_loss/truediv
Š
=training/Adam/gradients/loss/output_loss/truediv_grad/ReshapeReshape9training/Adam/gradients/loss/output_loss/truediv_grad/Sum;training/Adam/gradients/loss/output_loss/truediv_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*+
_class!
loc:@loss/output_loss/truediv*
Tshape0
ą
9training/Adam/gradients/loss/output_loss/truediv_grad/NegNegloss/output_loss/mul*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
?training/Adam/gradients/loss/output_loss/truediv_grad/RealDiv_1RealDiv9training/Adam/gradients/loss/output_loss/truediv_grad/Negloss/output_loss/Mean_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*+
_class!
loc:@loss/output_loss/truediv
˙
?training/Adam/gradients/loss/output_loss/truediv_grad/RealDiv_2RealDiv?training/Adam/gradients/loss/output_loss/truediv_grad/RealDiv_1loss/output_loss/Mean_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*+
_class!
loc:@loss/output_loss/truediv

9training/Adam/gradients/loss/output_loss/truediv_grad/mulMul<training/Adam/gradients/loss/output_loss/Mean_3_grad/truediv?training/Adam/gradients/loss/output_loss/truediv_grad/RealDiv_2*
T0*+
_class!
loc:@loss/output_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
;training/Adam/gradients/loss/output_loss/truediv_grad/Sum_1Sum9training/Adam/gradients/loss/output_loss/truediv_grad/mulMtraining/Adam/gradients/loss/output_loss/truediv_grad/BroadcastGradientArgs:1*
T0*+
_class!
loc:@loss/output_loss/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0
˘
?training/Adam/gradients/loss/output_loss/truediv_grad/Reshape_1Reshape;training/Adam/gradients/loss/output_loss/truediv_grad/Sum_1=training/Adam/gradients/loss/output_loss/truediv_grad/Shape_1*+
_class!
loc:@loss/output_loss/truediv*
Tshape0*
_output_shapes
: *
T0
ˇ
7training/Adam/gradients/loss/output_loss/mul_grad/ShapeShapeloss/output_loss/Mean_1*'
_class
loc:@loss/output_loss/mul*
out_type0*
_output_shapes
:*
T0
ˇ
9training/Adam/gradients/loss/output_loss/mul_grad/Shape_1Shapeoutput_sample_weights*
T0*'
_class
loc:@loss/output_loss/mul*
out_type0*
_output_shapes
:
ş
Gtraining/Adam/gradients/loss/output_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs7training/Adam/gradients/loss/output_loss/mul_grad/Shape9training/Adam/gradients/loss/output_loss/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0*'
_class
loc:@loss/output_loss/mul
é
5training/Adam/gradients/loss/output_loss/mul_grad/MulMul=training/Adam/gradients/loss/output_loss/truediv_grad/Reshapeoutput_sample_weights*'
_class
loc:@loss/output_loss/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ľ
5training/Adam/gradients/loss/output_loss/mul_grad/SumSum5training/Adam/gradients/loss/output_loss/mul_grad/MulGtraining/Adam/gradients/loss/output_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*'
_class
loc:@loss/output_loss/mul*
_output_shapes
:

9training/Adam/gradients/loss/output_loss/mul_grad/ReshapeReshape5training/Adam/gradients/loss/output_loss/mul_grad/Sum7training/Adam/gradients/loss/output_loss/mul_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*'
_class
loc:@loss/output_loss/mul*
Tshape0
í
7training/Adam/gradients/loss/output_loss/mul_grad/Mul_1Mulloss/output_loss/Mean_1=training/Adam/gradients/loss/output_loss/truediv_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*'
_class
loc:@loss/output_loss/mul
Ť
7training/Adam/gradients/loss/output_loss/mul_grad/Sum_1Sum7training/Adam/gradients/loss/output_loss/mul_grad/Mul_1Itraining/Adam/gradients/loss/output_loss/mul_grad/BroadcastGradientArgs:1*
T0*'
_class
loc:@loss/output_loss/mul*
_output_shapes
:*
	keep_dims( *

Tidx0

;training/Adam/gradients/loss/output_loss/mul_grad/Reshape_1Reshape7training/Adam/gradients/loss/output_loss/mul_grad/Sum_19training/Adam/gradients/loss/output_loss/mul_grad/Shape_1*
T0*'
_class
loc:@loss/output_loss/mul*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
:training/Adam/gradients/loss/output_loss/Mean_1_grad/ShapeShapeloss/output_loss/Mean*
T0**
_class 
loc:@loss/output_loss/Mean_1*
out_type0*
_output_shapes
:
§
9training/Adam/gradients/loss/output_loss/Mean_1_grad/SizeConst**
_class 
loc:@loss/output_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
ö
8training/Adam/gradients/loss/output_loss/Mean_1_grad/addAdd)loss/output_loss/Mean_1/reduction_indices9training/Adam/gradients/loss/output_loss/Mean_1_grad/Size**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:*
T0

8training/Adam/gradients/loss/output_loss/Mean_1_grad/modFloorMod8training/Adam/gradients/loss/output_loss/Mean_1_grad/add9training/Adam/gradients/loss/output_loss/Mean_1_grad/Size*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:
˛
<training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape_1Const**
_class 
loc:@loss/output_loss/Mean_1*
valueB:*
dtype0*
_output_shapes
:
Ž
@training/Adam/gradients/loss/output_loss/Mean_1_grad/range/startConst**
_class 
loc:@loss/output_loss/Mean_1*
value	B : *
dtype0*
_output_shapes
: 
Ž
@training/Adam/gradients/loss/output_loss/Mean_1_grad/range/deltaConst**
_class 
loc:@loss/output_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
Ö
:training/Adam/gradients/loss/output_loss/Mean_1_grad/rangeRange@training/Adam/gradients/loss/output_loss/Mean_1_grad/range/start9training/Adam/gradients/loss/output_loss/Mean_1_grad/Size@training/Adam/gradients/loss/output_loss/Mean_1_grad/range/delta**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:*

Tidx0
­
?training/Adam/gradients/loss/output_loss/Mean_1_grad/Fill/valueConst**
_class 
loc:@loss/output_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
Ł
9training/Adam/gradients/loss/output_loss/Mean_1_grad/FillFill<training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape_1?training/Adam/gradients/loss/output_loss/Mean_1_grad/Fill/value*
T0**
_class 
loc:@loss/output_loss/Mean_1*

index_type0*
_output_shapes
:

Btraining/Adam/gradients/loss/output_loss/Mean_1_grad/DynamicStitchDynamicStitch:training/Adam/gradients/loss/output_loss/Mean_1_grad/range8training/Adam/gradients/loss/output_loss/Mean_1_grad/mod:training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape9training/Adam/gradients/loss/output_loss/Mean_1_grad/Fill*
N*
_output_shapes
:*
T0**
_class 
loc:@loss/output_loss/Mean_1
Ź
>training/Adam/gradients/loss/output_loss/Mean_1_grad/Maximum/yConst*
_output_shapes
: **
_class 
loc:@loss/output_loss/Mean_1*
value	B :*
dtype0

<training/Adam/gradients/loss/output_loss/Mean_1_grad/MaximumMaximumBtraining/Adam/gradients/loss/output_loss/Mean_1_grad/DynamicStitch>training/Adam/gradients/loss/output_loss/Mean_1_grad/Maximum/y*
_output_shapes
:*
T0**
_class 
loc:@loss/output_loss/Mean_1

=training/Adam/gradients/loss/output_loss/Mean_1_grad/floordivFloorDiv:training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape<training/Adam/gradients/loss/output_loss/Mean_1_grad/Maximum*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
:
Č
<training/Adam/gradients/loss/output_loss/Mean_1_grad/ReshapeReshape9training/Adam/gradients/loss/output_loss/mul_grad/ReshapeBtraining/Adam/gradients/loss/output_loss/Mean_1_grad/DynamicStitch**
_class 
loc:@loss/output_loss/Mean_1*
Tshape0*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
Ä
9training/Adam/gradients/loss/output_loss/Mean_1_grad/TileTile<training/Adam/gradients/loss/output_loss/Mean_1_grad/Reshape=training/Adam/gradients/loss/output_loss/Mean_1_grad/floordiv*

Tmultiples0*
T0**
_class 
loc:@loss/output_loss/Mean_1*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
˝
<training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape_2Shapeloss/output_loss/Mean*
T0**
_class 
loc:@loss/output_loss/Mean_1*
out_type0*
_output_shapes
:
ż
<training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape_3Shapeloss/output_loss/Mean_1*
T0**
_class 
loc:@loss/output_loss/Mean_1*
out_type0*
_output_shapes
:
°
:training/Adam/gradients/loss/output_loss/Mean_1_grad/ConstConst**
_class 
loc:@loss/output_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
Ľ
9training/Adam/gradients/loss/output_loss/Mean_1_grad/ProdProd<training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape_2:training/Adam/gradients/loss/output_loss/Mean_1_grad/Const**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
˛
<training/Adam/gradients/loss/output_loss/Mean_1_grad/Const_1Const**
_class 
loc:@loss/output_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
Š
;training/Adam/gradients/loss/output_loss/Mean_1_grad/Prod_1Prod<training/Adam/gradients/loss/output_loss/Mean_1_grad/Shape_3<training/Adam/gradients/loss/output_loss/Mean_1_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0**
_class 
loc:@loss/output_loss/Mean_1
Ž
@training/Adam/gradients/loss/output_loss/Mean_1_grad/Maximum_1/yConst**
_class 
loc:@loss/output_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 

>training/Adam/gradients/loss/output_loss/Mean_1_grad/Maximum_1Maximum;training/Adam/gradients/loss/output_loss/Mean_1_grad/Prod_1@training/Adam/gradients/loss/output_loss/Mean_1_grad/Maximum_1/y*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: 

?training/Adam/gradients/loss/output_loss/Mean_1_grad/floordiv_1FloorDiv9training/Adam/gradients/loss/output_loss/Mean_1_grad/Prod>training/Adam/gradients/loss/output_loss/Mean_1_grad/Maximum_1*
T0**
_class 
loc:@loss/output_loss/Mean_1*
_output_shapes
: 
î
9training/Adam/gradients/loss/output_loss/Mean_1_grad/CastCast?training/Adam/gradients/loss/output_loss/Mean_1_grad/floordiv_1*

SrcT0**
_class 
loc:@loss/output_loss/Mean_1*
Truncate( *

DstT0*
_output_shapes
: 
Ą
<training/Adam/gradients/loss/output_loss/Mean_1_grad/truedivRealDiv9training/Adam/gradients/loss/output_loss/Mean_1_grad/Tile9training/Adam/gradients/loss/output_loss/Mean_1_grad/Cast*
T0**
_class 
loc:@loss/output_loss/Mean_1*-
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
ś
8training/Adam/gradients/loss/output_loss/Mean_grad/ShapeShapeloss/output_loss/Abs*
_output_shapes
:*
T0*(
_class
loc:@loss/output_loss/Mean*
out_type0
Ł
7training/Adam/gradients/loss/output_loss/Mean_grad/SizeConst*(
_class
loc:@loss/output_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
ę
6training/Adam/gradients/loss/output_loss/Mean_grad/addAdd'loss/output_loss/Mean/reduction_indices7training/Adam/gradients/loss/output_loss/Mean_grad/Size*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
ţ
6training/Adam/gradients/loss/output_loss/Mean_grad/modFloorMod6training/Adam/gradients/loss/output_loss/Mean_grad/add7training/Adam/gradients/loss/output_loss/Mean_grad/Size*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: 
§
:training/Adam/gradients/loss/output_loss/Mean_grad/Shape_1Const*(
_class
loc:@loss/output_loss/Mean*
valueB *
dtype0*
_output_shapes
: 
Ş
>training/Adam/gradients/loss/output_loss/Mean_grad/range/startConst*(
_class
loc:@loss/output_loss/Mean*
value	B : *
dtype0*
_output_shapes
: 
Ş
>training/Adam/gradients/loss/output_loss/Mean_grad/range/deltaConst*
_output_shapes
: *(
_class
loc:@loss/output_loss/Mean*
value	B :*
dtype0
Ě
8training/Adam/gradients/loss/output_loss/Mean_grad/rangeRange>training/Adam/gradients/loss/output_loss/Mean_grad/range/start7training/Adam/gradients/loss/output_loss/Mean_grad/Size>training/Adam/gradients/loss/output_loss/Mean_grad/range/delta*

Tidx0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:
Š
=training/Adam/gradients/loss/output_loss/Mean_grad/Fill/valueConst*(
_class
loc:@loss/output_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

7training/Adam/gradients/loss/output_loss/Mean_grad/FillFill:training/Adam/gradients/loss/output_loss/Mean_grad/Shape_1=training/Adam/gradients/loss/output_loss/Mean_grad/Fill/value*
T0*(
_class
loc:@loss/output_loss/Mean*

index_type0*
_output_shapes
: 

@training/Adam/gradients/loss/output_loss/Mean_grad/DynamicStitchDynamicStitch8training/Adam/gradients/loss/output_loss/Mean_grad/range6training/Adam/gradients/loss/output_loss/Mean_grad/mod8training/Adam/gradients/loss/output_loss/Mean_grad/Shape7training/Adam/gradients/loss/output_loss/Mean_grad/Fill*
T0*(
_class
loc:@loss/output_loss/Mean*
N*
_output_shapes
:
¨
<training/Adam/gradients/loss/output_loss/Mean_grad/Maximum/yConst*(
_class
loc:@loss/output_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

:training/Adam/gradients/loss/output_loss/Mean_grad/MaximumMaximum@training/Adam/gradients/loss/output_loss/Mean_grad/DynamicStitch<training/Adam/gradients/loss/output_loss/Mean_grad/Maximum/y*
_output_shapes
:*
T0*(
_class
loc:@loss/output_loss/Mean

;training/Adam/gradients/loss/output_loss/Mean_grad/floordivFloorDiv8training/Adam/gradients/loss/output_loss/Mean_grad/Shape:training/Adam/gradients/loss/output_loss/Mean_grad/Maximum*
T0*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
:
Ň
:training/Adam/gradients/loss/output_loss/Mean_grad/ReshapeReshape<training/Adam/gradients/loss/output_loss/Mean_1_grad/truediv@training/Adam/gradients/loss/output_loss/Mean_grad/DynamicStitch*
T0*(
_class
loc:@loss/output_loss/Mean*
Tshape0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
É
7training/Adam/gradients/loss/output_loss/Mean_grad/TileTile:training/Adam/gradients/loss/output_loss/Mean_grad/Reshape;training/Adam/gradients/loss/output_loss/Mean_grad/floordiv*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0*(
_class
loc:@loss/output_loss/Mean
¸
:training/Adam/gradients/loss/output_loss/Mean_grad/Shape_2Shapeloss/output_loss/Abs*
T0*(
_class
loc:@loss/output_loss/Mean*
out_type0*
_output_shapes
:
š
:training/Adam/gradients/loss/output_loss/Mean_grad/Shape_3Shapeloss/output_loss/Mean*
T0*(
_class
loc:@loss/output_loss/Mean*
out_type0*
_output_shapes
:
Ź
8training/Adam/gradients/loss/output_loss/Mean_grad/ConstConst*(
_class
loc:@loss/output_loss/Mean*
valueB: *
dtype0*
_output_shapes
:

7training/Adam/gradients/loss/output_loss/Mean_grad/ProdProd:training/Adam/gradients/loss/output_loss/Mean_grad/Shape_28training/Adam/gradients/loss/output_loss/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*(
_class
loc:@loss/output_loss/Mean
Ž
:training/Adam/gradients/loss/output_loss/Mean_grad/Const_1Const*
_output_shapes
:*(
_class
loc:@loss/output_loss/Mean*
valueB: *
dtype0
Ą
9training/Adam/gradients/loss/output_loss/Mean_grad/Prod_1Prod:training/Adam/gradients/loss/output_loss/Mean_grad/Shape_3:training/Adam/gradients/loss/output_loss/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*(
_class
loc:@loss/output_loss/Mean
Ş
>training/Adam/gradients/loss/output_loss/Mean_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *(
_class
loc:@loss/output_loss/Mean*
value	B :

<training/Adam/gradients/loss/output_loss/Mean_grad/Maximum_1Maximum9training/Adam/gradients/loss/output_loss/Mean_grad/Prod_1>training/Adam/gradients/loss/output_loss/Mean_grad/Maximum_1/y*
_output_shapes
: *
T0*(
_class
loc:@loss/output_loss/Mean

=training/Adam/gradients/loss/output_loss/Mean_grad/floordiv_1FloorDiv7training/Adam/gradients/loss/output_loss/Mean_grad/Prod<training/Adam/gradients/loss/output_loss/Mean_grad/Maximum_1*(
_class
loc:@loss/output_loss/Mean*
_output_shapes
: *
T0
č
7training/Adam/gradients/loss/output_loss/Mean_grad/CastCast=training/Adam/gradients/loss/output_loss/Mean_grad/floordiv_1*

SrcT0*(
_class
loc:@loss/output_loss/Mean*
Truncate( *

DstT0*
_output_shapes
: 

:training/Adam/gradients/loss/output_loss/Mean_grad/truedivRealDiv7training/Adam/gradients/loss/output_loss/Mean_grad/Tile7training/Adam/gradients/loss/output_loss/Mean_grad/Cast*
T0*(
_class
loc:@loss/output_loss/Mean*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
š
6training/Adam/gradients/loss/output_loss/Abs_grad/SignSignloss/output_loss/sub*
T0*'
_class
loc:@loss/output_loss/Abs*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ

5training/Adam/gradients/loss/output_loss/Abs_grad/mulMul:training/Adam/gradients/loss/output_loss/Mean_grad/truediv6training/Adam/gradients/loss/output_loss/Abs_grad/Sign*
T0*'
_class
loc:@loss/output_loss/Abs*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
ľ
7training/Adam/gradients/loss/output_loss/sub_grad/ShapeShapeoutput/ResizeBilinear*
T0*'
_class
loc:@loss/output_loss/sub*
out_type0*
_output_shapes
:
Ż
9training/Adam/gradients/loss/output_loss/sub_grad/Shape_1Shapeoutput_target*
T0*'
_class
loc:@loss/output_loss/sub*
out_type0*
_output_shapes
:
ş
Gtraining/Adam/gradients/loss/output_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs7training/Adam/gradients/loss/output_loss/sub_grad/Shape9training/Adam/gradients/loss/output_loss/sub_grad/Shape_1*
T0*'
_class
loc:@loss/output_loss/sub*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ľ
5training/Adam/gradients/loss/output_loss/sub_grad/SumSum5training/Adam/gradients/loss/output_loss/Abs_grad/mulGtraining/Adam/gradients/loss/output_loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*'
_class
loc:@loss/output_loss/sub
§
9training/Adam/gradients/loss/output_loss/sub_grad/ReshapeReshape5training/Adam/gradients/loss/output_loss/sub_grad/Sum7training/Adam/gradients/loss/output_loss/sub_grad/Shape*'
_class
loc:@loss/output_loss/sub*
Tshape0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
T0
Š
7training/Adam/gradients/loss/output_loss/sub_grad/Sum_1Sum5training/Adam/gradients/loss/output_loss/Abs_grad/mulItraining/Adam/gradients/loss/output_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*'
_class
loc:@loss/output_loss/sub*
_output_shapes
:
Á
5training/Adam/gradients/loss/output_loss/sub_grad/NegNeg7training/Adam/gradients/loss/output_loss/sub_grad/Sum_1*
T0*'
_class
loc:@loss/output_loss/sub*
_output_shapes
:
Ä
;training/Adam/gradients/loss/output_loss/sub_grad/Reshape_1Reshape5training/Adam/gradients/loss/output_loss/sub_grad/Neg9training/Adam/gradients/loss/output_loss/sub_grad/Shape_1*
T0*'
_class
loc:@loss/output_loss/sub*
Tshape0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ś
Etraining/Adam/gradients/output/ResizeBilinear_grad/ResizeBilinearGradResizeBilinearGrad9training/Adam/gradients/loss/output_loss/sub_grad/Reshapeoutput/DepthToSpace*(
_class
loc:@output/ResizeBilinear*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
align_corners( *
T0
Ą
=training/Adam/gradients/output/DepthToSpace_grad/SpaceToDepthSpaceToDepthEtraining/Adam/gradients/output/ResizeBilinear_grad/ResizeBilinearGrad*&
_class
loc:@output/DepthToSpace*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨*

block_size*
T0
č
9training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGradBiasAddGrad=training/Adam/gradients/output/DepthToSpace_grad/SpaceToDepth*
T0*#
_class
loc:@conv2d_6/BiasAdd*
data_formatNHWC*
_output_shapes
:
Ů
8training/Adam/gradients/conv2d_6/convolution_grad/ShapeNShapeNdropout_3/Identityconv2d_6/kernel/read*
N* 
_output_shapes
::*
T0*'
_class
loc:@conv2d_6/convolution*
out_type0
ž
Etraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_6/convolution_grad/ShapeNconv2d_6/kernel/read=training/Adam/gradients/output/DepthToSpace_grad/SpaceToDepth*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
	dilations
*
T0*'
_class
loc:@conv2d_6/convolution*
data_formatNHWC*
strides

ľ
Ftraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdropout_3/Identity:training/Adam/gradients/conv2d_6/convolution_grad/ShapeN:1=training/Adam/gradients/output/DepthToSpace_grad/SpaceToDepth*&
_output_shapes
:@*
	dilations
*
T0*'
_class
loc:@conv2d_6/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
¤
Btraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGradLeakyReluGradEtraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputconv2d_5/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_6/LeakyRelu*
alpha%>*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@
í
9training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_5/BiasAdd*
data_formatNHWC*
_output_shapes
:@
Ţ
8training/Adam/gradients/conv2d_5/convolution_grad/ShapeNShapeNleaky_re_lu_5/LeakyReluconv2d_5/kernel/read*
N* 
_output_shapes
::*
T0*'
_class
loc:@conv2d_5/convolution*
out_type0
Ă
Etraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_5/convolution_grad/ShapeNconv2d_5/kernel/readBtraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
	dilations
*
T0*'
_class
loc:@conv2d_5/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ż
Ftraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterleaky_re_lu_5/LeakyRelu:training/Adam/gradients/conv2d_5/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_6/LeakyRelu_grad/LeakyReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_5/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@@
¤
Btraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGradLeakyReluGradEtraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropInputconv2d_4/BiasAdd*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
T0**
_class 
loc:@leaky_re_lu_5/LeakyRelu*
alpha%>
í
9training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_4/BiasAdd*
data_formatNHWC*
_output_shapes
:@
ě
8training/Adam/gradients/conv2d_4/convolution_grad/ShapeNShapeN%up_sampling2d_1/ResizeNearestNeighborconv2d_4/kernel/read* 
_output_shapes
::*
T0*'
_class
loc:@conv2d_4/convolution*
out_type0*
N
Ä
Etraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_4/convolution_grad/ShapeNconv2d_4/kernel/readBtraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
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
Î
Ftraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_1/ResizeNearestNeighbor:training/Adam/gradients/conv2d_4/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_5/LeakyRelu_grad/LeakyReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_4/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@
ě
atraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*
valueB"    *
dtype0*
_output_shapes
:
Ż
\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradEtraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputatraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
align_corners( *
T0

@training/Adam/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGraddropout_2/Identitymax_pooling2d_2/MaxPool\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*
ksize
*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0**
_class 
loc:@max_pooling2d_2/MaxPool*
data_formatNHWC*
strides

 
Btraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGradLeakyReluGrad@training/Adam/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradconv2d_3/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_4/LeakyRelu*
alpha%>*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
î
9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_3/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ţ
8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNShapeNleaky_re_lu_3/LeakyReluconv2d_3/kernel/read*
T0*'
_class
loc:@conv2d_3/convolution*
out_type0*
N* 
_output_shapes
::
Ä
Etraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNconv2d_3/kernel/readBtraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_3/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
Á
Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterleaky_re_lu_3/LeakyRelu:training/Adam/gradients/conv2d_3/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_4/LeakyRelu_grad/LeakyReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0*'
_class
loc:@conv2d_3/convolution
Ľ
Btraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGradLeakyReluGradEtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputconv2d_2/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_3/LeakyRelu*
alpha%>*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
î
9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_2/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ţ
8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNShapeNmax_pooling2d_1/MaxPoolconv2d_2/kernel/read*
T0*'
_class
loc:@conv2d_2/convolution*
out_type0*
N* 
_output_shapes
::
Ă
Etraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNconv2d_2/kernel/readBtraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGrad*
T0*'
_class
loc:@conv2d_2/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
	dilations

Ŕ
Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_1/MaxPool:training/Adam/gradients/conv2d_2/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@
ý
@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGraddropout_1/Identitymax_pooling2d_1/MaxPoolEtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput*
ksize
*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0**
_class 
loc:@max_pooling2d_1/MaxPool*
data_formatNHWC*
strides


Btraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGradLeakyReluGrad@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradconv2d_1/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_2/LeakyRelu*
alpha%>*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
í
9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
T0*#
_class
loc:@conv2d_1/BiasAdd*
data_formatNHWC*
_output_shapes
:@
Ţ
8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNShapeNleaky_re_lu_1/LeakyReluconv2d_1/kernel/read*'
_class
loc:@conv2d_1/convolution*
out_type0*
N* 
_output_shapes
::*
T0
Ă
Etraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNconv2d_1/kernel/readBtraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
T0*'
_class
loc:@conv2d_1/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
	dilations

ż
Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterleaky_re_lu_1/LeakyRelu:training/Adam/gradients/conv2d_1/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
paddingSAME*&
_output_shapes
:@@*
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ą
Btraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGradLeakyReluGradEtraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputinput/BiasAdd*
T0**
_class 
loc:@leaky_re_lu_1/LeakyRelu*
alpha%>*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
ç
6training/Adam/gradients/input/BiasAdd_grad/BiasAddGradBiasAddGradBtraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
T0* 
_class
loc:@input/BiasAdd*
data_formatNHWC*
_output_shapes
:@
É
5training/Adam/gradients/input/convolution_grad/ShapeNShapeNinput_inputinput/kernel/read*
T0*$
_class
loc:@input/convolution*
out_type0*
N* 
_output_shapes
::
ˇ
Btraining/Adam/gradients/input/convolution_grad/Conv2DBackpropInputConv2DBackpropInput5training/Adam/gradients/input/convolution_grad/ShapeNinput/kernel/readBtraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
	dilations
*
T0*$
_class
loc:@input/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Ş
Ctraining/Adam/gradients/input/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterinput_input7training/Adam/gradients/input/convolution_grad/ShapeN:1Btraining/Adam/gradients/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
T0*$
_class
loc:@input/convolution*
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
training/Adam/AssignAdd/valueConst*
dtype0	*
_output_shapes
: *
value	B	 R
Ź
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0	*"
_class
loc:@Adam/iterations
p
training/Adam/CastCastAdam/iterations/read*

SrcT0	*
Truncate( *

DstT0*
_output_shapes
: 
X
training/Adam/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
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
training/Adam/Const_1Const*
_output_shapes
: *
valueB
 *  *
dtype0
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
_output_shapes
: *
T0

training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
_output_shapes
: *
T0
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
#training/Adam/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"         @   
^
training/Adam/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zerosFill#training/Adam/zeros/shape_as_tensortraining/Adam/zeros/Const*

index_type0*&
_output_shapes
:@*
T0

training/Adam/Variable
VariableV2*
dtype0*
	container *&
_output_shapes
:@*
shape:@*
shared_name 
Ů
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*)
_class
loc:@training/Adam/Variable*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
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
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
Ő
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:@
~
%training/Adam/zeros_2/shape_as_tensorConst*%
valueB"      @   @   *
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
¤
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*&
_output_shapes
:@@*
T0*

index_type0

training/Adam/Variable_2
VariableV2*
shape:@@*
shared_name *
dtype0*
	container *&
_output_shapes
:@@
á
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*&
_output_shapes
:@@
Ą
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*&
_output_shapes
:@@*
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
VariableV2*
shape:@*
shared_name *
dtype0*
	container *
_output_shapes
:@
Ő
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0

training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
:@*
T0
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
Ľ
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
â
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*'
_output_shapes
:@
˘
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
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ö
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes	
:*
use_locking(
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
training/Adam/zeros_6/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ś
training/Adam/zeros_6Fill%training/Adam/zeros_6/shape_as_tensortraining/Adam/zeros_6/Const*

index_type0*(
_output_shapes
:*
T0
 
training/Adam/Variable_6
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:*
shape:
ă
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Ł
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*(
_output_shapes
:*
T0*+
_class!
loc:@training/Adam/Variable_6
d
training/Adam/zeros_7Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_7
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ö
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
_output_shapes	
:*
T0*+
_class!
loc:@training/Adam/Variable_7
~
%training/Adam/zeros_8/shape_as_tensorConst*%
valueB"         @   *
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
Ľ
training/Adam/zeros_8Fill%training/Adam/zeros_8/shape_as_tensortraining/Adam/zeros_8/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_8
VariableV2*
shared_name *
dtype0*
	container *'
_output_shapes
:@*
shape:@
â
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*'
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(
˘
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
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
Ő
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
§
training/Adam/zeros_10Fill&training/Adam/zeros_10/shape_as_tensortraining/Adam/zeros_10/Const*
T0*

index_type0*&
_output_shapes
:@@

training/Adam/Variable_10
VariableV2*
shape:@@*
shared_name *
dtype0*
	container *&
_output_shapes
:@@
ĺ
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*&
_output_shapes
:@@*
use_locking(
¤
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
VariableV2*
	container *
_output_shapes
:@*
shape:@*
shared_name *
dtype0
Ů
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:@
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
dtype0*
	container *&
_output_shapes
:@*
shape:@*
shared_name 
ĺ
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*&
_output_shapes
:@
¤
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
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ů
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
training/Adam/zeros_14/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
§
training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*
T0*

index_type0*&
_output_shapes
:@

training/Adam/Variable_14
VariableV2*
shared_name *
dtype0*
	container *&
_output_shapes
:@*
shape:@
ĺ
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*&
_output_shapes
:@
¤
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*&
_output_shapes
:@
c
training/Adam/zeros_15Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_15
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:@*
shape:@
Ů
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_15

&training/Adam/zeros_16/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*

index_type0*&
_output_shapes
:@@*
T0

training/Adam/Variable_16
VariableV2*
shape:@@*
shared_name *
dtype0*
	container *&
_output_shapes
:@@
ĺ
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*&
_output_shapes
:@@
¤
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
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
Ů
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_17
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
¨
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*'
_output_shapes
:@*
T0*

index_type0

training/Adam/Variable_18
VariableV2*
dtype0*
	container *'
_output_shapes
:@*
shape:@*
shared_name 
ć
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*'
_output_shapes
:@*
use_locking(
Ľ
training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*
T0*,
_class"
 loc:@training/Adam/Variable_18*'
_output_shapes
:@
e
training/Adam/zeros_19Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_19
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
Ú
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_19

&training/Adam/zeros_20/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_20/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Š
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_20
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
ç
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(
Ś
training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
T0*,
_class"
 loc:@training/Adam/Variable_20*(
_output_shapes
:
e
training/Adam/zeros_21Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_21
VariableV2*
	container *
_output_shapes	
:*
shape:*
shared_name *
dtype0
Ú
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
&training/Adam/zeros_22/shape_as_tensorConst*
_output_shapes
:*%
valueB"         @   *
dtype0
a
training/Adam/zeros_22/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¨
training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_22
VariableV2*
shared_name *
dtype0*
	container *'
_output_shapes
:@*
shape:@
ć
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*'
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(
Ľ
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
Ů
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
&training/Adam/zeros_24/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_24/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
training/Adam/zeros_24Fill&training/Adam/zeros_24/shape_as_tensortraining/Adam/zeros_24/Const*&
_output_shapes
:@@*
T0*

index_type0

training/Adam/Variable_24
VariableV2*
dtype0*
	container *&
_output_shapes
:@@*
shape:@@*
shared_name 
ĺ
 training/Adam/Variable_24/AssignAssigntraining/Adam/Variable_24training/Adam/zeros_24*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24
¤
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
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:@*
shape:@
Ů
 training/Adam/Variable_25/AssignAssigntraining/Adam/Variable_25training/Adam/zeros_25*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_25/readIdentitytraining/Adam/Variable_25*,
_class"
 loc:@training/Adam/Variable_25*
_output_shapes
:@*
T0
{
training/Adam/zeros_26Const*&
_output_shapes
:@*%
valueB@*    *
dtype0

training/Adam/Variable_26
VariableV2*
dtype0*
	container *&
_output_shapes
:@*
shape:@*
shared_name 
ĺ
 training/Adam/Variable_26/AssignAssigntraining/Adam/Variable_26training/Adam/zeros_26*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_26*
validate_shape(*&
_output_shapes
:@
¤
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
Ů
 training/Adam/Variable_27/AssignAssigntraining/Adam/Variable_27training/Adam/zeros_27*,
_class"
 loc:@training/Adam/Variable_27*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_27/readIdentitytraining/Adam/Variable_27*
T0*,
_class"
 loc:@training/Adam/Variable_27*
_output_shapes
:
p
&training/Adam/zeros_28/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
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
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
Ů
 training/Adam/Variable_28/AssignAssigntraining/Adam/Variable_28training/Adam/zeros_28*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_28*
validate_shape(*
_output_shapes
:
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
training/Adam/zeros_29Fill&training/Adam/zeros_29/shape_as_tensortraining/Adam/zeros_29/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_29
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ů
 training/Adam/Variable_29/AssignAssigntraining/Adam/Variable_29training/Adam/zeros_29*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_29
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
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ů
 training/Adam/Variable_30/AssignAssigntraining/Adam/Variable_30training/Adam/zeros_30*,
_class"
 loc:@training/Adam/Variable_30*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_30/readIdentitytraining/Adam/Variable_30*
T0*,
_class"
 loc:@training/Adam/Variable_30*
_output_shapes
:
p
&training/Adam/zeros_31/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_31/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_31Fill&training/Adam/zeros_31/shape_as_tensortraining/Adam/zeros_31/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_31
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
Ů
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
training/Adam/zeros_32Fill&training/Adam/zeros_32/shape_as_tensortraining/Adam/zeros_32/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_32
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ů
 training/Adam/Variable_32/AssignAssigntraining/Adam/Variable_32training/Adam/zeros_32*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_32*
validate_shape(*
_output_shapes
:

training/Adam/Variable_32/readIdentitytraining/Adam/Variable_32*,
_class"
 loc:@training/Adam/Variable_32*
_output_shapes
:*
T0
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
training/Adam/zeros_33Fill&training/Adam/zeros_33/shape_as_tensortraining/Adam/zeros_33/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_33
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
Ů
 training/Adam/Variable_33/AssignAssigntraining/Adam/Variable_33training/Adam/zeros_33*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_33*
validate_shape(*
_output_shapes
:
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
training/Adam/zeros_34/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_34Fill&training/Adam/zeros_34/shape_as_tensortraining/Adam/zeros_34/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_34
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
Ů
 training/Adam/Variable_34/AssignAssigntraining/Adam/Variable_34training/Adam/zeros_34*,
_class"
 loc:@training/Adam/Variable_34*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_34/readIdentitytraining/Adam/Variable_34*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_34
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
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
Ů
 training/Adam/Variable_35/AssignAssigntraining/Adam/Variable_35training/Adam/zeros_35*,
_class"
 loc:@training/Adam/Variable_35*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
Ů
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
&training/Adam/zeros_37/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
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
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ů
 training/Adam/Variable_37/AssignAssigntraining/Adam/Variable_37training/Adam/zeros_37*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_37*
validate_shape(*
_output_shapes
:
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
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ů
 training/Adam/Variable_38/AssignAssigntraining/Adam/Variable_38training/Adam/zeros_38*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_38*
validate_shape(*
_output_shapes
:

training/Adam/Variable_38/readIdentitytraining/Adam/Variable_38*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_38
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
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
Ů
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
&training/Adam/zeros_40/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_40/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_40Fill&training/Adam/zeros_40/shape_as_tensortraining/Adam/zeros_40/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_40
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
Ů
 training/Adam/Variable_40/AssignAssigntraining/Adam/Variable_40training/Adam/zeros_40*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_40*
validate_shape(*
_output_shapes
:

training/Adam/Variable_40/readIdentitytraining/Adam/Variable_40*,
_class"
 loc:@training/Adam/Variable_40*
_output_shapes
:*
T0
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
Ů
 training/Adam/Variable_41/AssignAssigntraining/Adam/Variable_41training/Adam/zeros_41*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_41*
validate_shape(*
_output_shapes
:

training/Adam/Variable_41/readIdentitytraining/Adam/Variable_41*
T0*,
_class"
 loc:@training/Adam/Variable_41*
_output_shapes
:
z
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*&
_output_shapes
:@*
T0
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
Ľ
training/Adam/mul_2Multraining/Adam/sub_2Ctraining/Adam/gradients/input/convolution_grad/Conv2DBackpropFilter*
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
training/Adam/sub_3/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/SquareSquareCtraining/Adam/gradients/input/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
v
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*&
_output_shapes
:@
u
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*&
_output_shapes
:@*
T0
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
training/Adam/add_3/yConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
x
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*&
_output_shapes
:@*
T0
}
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*&
_output_shapes
:@*
T0
w
training/Adam/sub_4Subinput/kernel/readtraining/Adam/truediv_1*
T0*&
_output_shapes
:@
Đ
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*&
_output_shapes
:@
Ř
training/Adam/Assign_1Assigntraining/Adam/Variable_14training/Adam/add_2*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14
ž
training/Adam/Assign_2Assigninput/kerneltraining/Adam/sub_4*
T0*
_class
loc:@input/kernel*
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

training/Adam/mul_7Multraining/Adam/sub_56training/Adam/gradients/input/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
:@
q
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_15/read*
_output_shapes
:@*
T0
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
}
training/Adam/Square_1Square6training/Adam/gradients/input/BiasAdd_grad/BiasAddGrad*
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
training/Adam/Const_5Const*
valueB
 *  *
dtype0*
_output_shapes
: 
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
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
:@
Z
training/Adam/add_6/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
_output_shapes
:@*
T0
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
_output_shapes
:@*
T0
i
training/Adam/sub_7Subinput/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes
:@
Ę
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(
Ě
training/Adam/Assign_4Assigntraining/Adam/Variable_15training/Adam/add_5*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:@
Ž
training/Adam/Assign_5Assign
input/biastraining/Adam/sub_7*
use_locking(*
T0*
_class
loc:@input/bias*
validate_shape(*
_output_shapes
:@
}
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*&
_output_shapes
:@@*
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
Š
training/Adam/mul_12Multraining/Adam/sub_8Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
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
training/Adam/Square_2SquareFtraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
y
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*&
_output_shapes
:@@
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
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*
T0*&
_output_shapes
:@@

training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*&
_output_shapes
:@@*
T0
l
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*&
_output_shapes
:@@
Z
training/Adam/add_9/yConst*
valueB
 *żÖ3*
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
training/Adam/sub_10Subconv2d_1/kernel/readtraining/Adam/truediv_3*&
_output_shapes
:@@*
T0
Ö
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*&
_output_shapes
:@@
Ř
training/Adam/Assign_7Assigntraining/Adam/Variable_16training/Adam/add_8*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*&
_output_shapes
:@@
Ĺ
training/Adam/Assign_8Assignconv2d_1/kerneltraining/Adam/sub_10*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@@*
use_locking(
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
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_17Multraining/Adam/sub_119training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
:@
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_17/read*
T0*
_output_shapes
:@
[
training/Adam/sub_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_3Square9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
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
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
_output_shapes
:@*
T0
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
 *żÖ3*
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
training/Adam/sub_13Subconv2d_1/bias/readtraining/Adam/truediv_4*
_output_shapes
:@*
T0
Ë
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:@
Î
training/Adam/Assign_10Assigntraining/Adam/Variable_17training/Adam/add_11*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17
ś
training/Adam/Assign_11Assignconv2d_1/biastraining/Adam/sub_13*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias
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
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ť
training/Adam/mul_22Multraining/Adam/sub_14Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*'
_output_shapes
:@*
T0
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
training/Adam/Square_4SquareFtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
{
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*'
_output_shapes
:@
y
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*'
_output_shapes
:@*
T0
v
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*'
_output_shapes
:@
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
training/Adam/add_15/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
{
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*'
_output_shapes
:@

training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*'
_output_shapes
:@
|
training/Adam/sub_16Subconv2d_2/kernel/readtraining/Adam/truediv_5*
T0*'
_output_shapes
:@
Ů
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*'
_output_shapes
:@
Ű
training/Adam/Assign_13Assigntraining/Adam/Variable_18training/Adam/add_14*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18
Ç
training/Adam/Assign_14Assignconv2d_2/kerneltraining/Adam/sub_16*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*'
_output_shapes
:@
r
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
T0*
_output_shapes	
:
[
training/Adam/sub_17/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_27Multraining/Adam/sub_179training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
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
training/Adam/Square_5Square9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
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
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
_output_shapes	
:*
T0
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
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
T0*
_output_shapes	
:
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
 *żÖ3*
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
training/Adam/sub_19Subconv2d_2/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes	
:
Í
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes	
:*
use_locking(
Ď
training/Adam/Assign_16Assigntraining/Adam/Variable_19training/Adam/add_17*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_19
ˇ
training/Adam/Assign_17Assignconv2d_2/biastraining/Adam/sub_19*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias

training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*(
_output_shapes
:
[
training/Adam/sub_20/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ź
training/Adam/mul_32Multraining/Adam/sub_20Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
z
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*(
_output_shapes
:*
T0
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
training/Adam/Square_6SquareFtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
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
training/Adam/Const_14Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_15Const*
_output_shapes
: *
valueB
 *  *
dtype0
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
training/Adam/add_21/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
|
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
T0*(
_output_shapes
:

training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*(
_output_shapes
:
}
training/Adam/sub_22Subconv2d_3/kernel/readtraining/Adam/truediv_7*
T0*(
_output_shapes
:
Ú
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_19Assigntraining/Adam/Variable_20training/Adam/add_20*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(*(
_output_shapes
:*
use_locking(
Č
training/Adam/Assign_20Assignconv2d_3/kerneltraining/Adam/sub_22*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(*(
_output_shapes
:
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
training/Adam/mul_37Multraining/Adam/sub_239training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
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
training/Adam/Square_7Square9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
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
training/Adam/Const_17Const*
dtype0*
_output_shapes
: *
valueB
 *  

%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_17*
T0*
_output_shapes	
:

training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_16*
T0*
_output_shapes	
:
a
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
_output_shapes	
:*
T0
[
training/Adam/add_24/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
o
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
_output_shapes	
:*
T0
t
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
_output_shapes	
:*
T0
n
training/Adam/sub_25Subconv2d_3/bias/readtraining/Adam/truediv_8*
T0*
_output_shapes	
:
Í
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes	
:*
use_locking(
Ď
training/Adam/Assign_22Assigntraining/Adam/Variable_21training/Adam/add_23*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes	
:*
use_locking(
ˇ
training/Adam/Assign_23Assignconv2d_3/biastraining/Adam/sub_25*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes	
:
~
training/Adam/mul_41MulAdam/beta_1/readtraining/Adam/Variable_8/read*'
_output_shapes
:@*
T0
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
Ť
training/Adam/mul_42Multraining/Adam/sub_26Ftraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*
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
training/Adam/Square_8SquareFtraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*
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
training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_18*'
_output_shapes
:@*
T0
m
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
T0*'
_output_shapes
:@
[
training/Adam/add_27/yConst*
_output_shapes
: *
valueB
 *żÖ3*
dtype0
{
training/Adam/add_27Addtraining/Adam/Sqrt_9training/Adam/add_27/y*'
_output_shapes
:@*
T0

training/Adam/truediv_9RealDivtraining/Adam/mul_45training/Adam/add_27*'
_output_shapes
:@*
T0
|
training/Adam/sub_28Subconv2d_4/kernel/readtraining/Adam/truediv_9*'
_output_shapes
:@*
T0
Ů
training/Adam/Assign_24Assigntraining/Adam/Variable_8training/Adam/add_25*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0
Ű
training/Adam/Assign_25Assigntraining/Adam/Variable_22training/Adam/add_26*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*'
_output_shapes
:@*
use_locking(
Ç
training/Adam/Assign_26Assignconv2d_4/kerneltraining/Adam/sub_28*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*'
_output_shapes
:@
q
training/Adam/mul_46MulAdam/beta_1/readtraining/Adam/Variable_9/read*
_output_shapes
:@*
T0
[
training/Adam/sub_29/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_29Subtraining/Adam/sub_29/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_47Multraining/Adam/sub_299training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
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
training/Adam/sub_30Subtraining/Adam/sub_30/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_9Square9training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
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
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
_output_shapes
:@*
T0
[
training/Adam/add_30/yConst*
valueB
 *żÖ3*
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
training/Adam/sub_31Subconv2d_4/bias/readtraining/Adam/truediv_10*
T0*
_output_shapes
:@
Ě
training/Adam/Assign_27Assigntraining/Adam/Variable_9training/Adam/add_28*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:@
Î
training/Adam/Assign_28Assigntraining/Adam/Variable_23training/Adam/add_29*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes
:@
ś
training/Adam/Assign_29Assignconv2d_4/biastraining/Adam/sub_31*
use_locking(*
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(*
_output_shapes
:@
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
training/Adam/sub_32Subtraining/Adam/sub_32/xAdam/beta_1/read*
_output_shapes
: *
T0
Ş
training/Adam/mul_52Multraining/Adam/sub_32Ftraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*
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
training/Adam/Square_10SquareFtraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
{
training/Adam/mul_54Multraining/Adam/sub_33training/Adam/Square_10*
T0*&
_output_shapes
:@@
x
training/Adam/add_32Addtraining/Adam/mul_53training/Adam/mul_54*
T0*&
_output_shapes
:@@
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
&training/Adam/clip_by_value_11/MinimumMinimumtraining/Adam/add_32training/Adam/Const_23*&
_output_shapes
:@@*
T0

training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_22*
T0*&
_output_shapes
:@@
n
training/Adam/Sqrt_11Sqrttraining/Adam/clip_by_value_11*
T0*&
_output_shapes
:@@
[
training/Adam/add_33/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
{
training/Adam/add_33Addtraining/Adam/Sqrt_11training/Adam/add_33/y*
T0*&
_output_shapes
:@@

training/Adam/truediv_11RealDivtraining/Adam/mul_55training/Adam/add_33*
T0*&
_output_shapes
:@@
|
training/Adam/sub_34Subconv2d_5/kernel/readtraining/Adam/truediv_11*
T0*&
_output_shapes
:@@
Ú
training/Adam/Assign_30Assigntraining/Adam/Variable_10training/Adam/add_31*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10
Ú
training/Adam/Assign_31Assigntraining/Adam/Variable_24training/Adam/add_32*&
_output_shapes
:@@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(
Ć
training/Adam/Assign_32Assignconv2d_5/kerneltraining/Adam/sub_34*
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel*
validate_shape(*&
_output_shapes
:@@
r
training/Adam/mul_56MulAdam/beta_1/readtraining/Adam/Variable_11/read*
T0*
_output_shapes
:@
[
training/Adam/sub_35/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_35Subtraining/Adam/sub_35/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_57Multraining/Adam/sub_359training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
l
training/Adam/add_34Addtraining/Adam/mul_56training/Adam/mul_57*
T0*
_output_shapes
:@
r
training/Adam/mul_58MulAdam/beta_2/readtraining/Adam/Variable_25/read*
_output_shapes
:@*
T0
[
training/Adam/sub_36/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_36Subtraining/Adam/sub_36/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_11Square9training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
o
training/Adam/mul_59Multraining/Adam/sub_36training/Adam/Square_11*
_output_shapes
:@*
T0
l
training/Adam/add_35Addtraining/Adam/mul_58training/Adam/mul_59*
_output_shapes
:@*
T0
i
training/Adam/mul_60Multraining/Adam/multraining/Adam/add_34*
T0*
_output_shapes
:@
[
training/Adam/Const_24Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_25Const*
dtype0*
_output_shapes
: *
valueB
 *  

&training/Adam/clip_by_value_12/MinimumMinimumtraining/Adam/add_35training/Adam/Const_25*
_output_shapes
:@*
T0

training/Adam/clip_by_value_12Maximum&training/Adam/clip_by_value_12/Minimumtraining/Adam/Const_24*
_output_shapes
:@*
T0
b
training/Adam/Sqrt_12Sqrttraining/Adam/clip_by_value_12*
T0*
_output_shapes
:@
[
training/Adam/add_36/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
o
training/Adam/add_36Addtraining/Adam/Sqrt_12training/Adam/add_36/y*
T0*
_output_shapes
:@
t
training/Adam/truediv_12RealDivtraining/Adam/mul_60training/Adam/add_36*
_output_shapes
:@*
T0
n
training/Adam/sub_37Subconv2d_5/bias/readtraining/Adam/truediv_12*
T0*
_output_shapes
:@
Î
training/Adam/Assign_33Assigntraining/Adam/Variable_11training/Adam/add_34*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:@
Î
training/Adam/Assign_34Assigntraining/Adam/Variable_25training/Adam/add_35*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(*
_output_shapes
:@
ś
training/Adam/Assign_35Assignconv2d_5/biastraining/Adam/sub_37*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias
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
training/Adam/sub_38Subtraining/Adam/sub_38/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ş
training/Adam/mul_62Multraining/Adam/sub_38Ftraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
x
training/Adam/add_37Addtraining/Adam/mul_61training/Adam/mul_62*
T0*&
_output_shapes
:@
~
training/Adam/mul_63MulAdam/beta_2/readtraining/Adam/Variable_26/read*&
_output_shapes
:@*
T0
[
training/Adam/sub_39/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_39Subtraining/Adam/sub_39/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_12SquareFtraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
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
training/Adam/mul_65Multraining/Adam/multraining/Adam/add_37*&
_output_shapes
:@*
T0
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
&training/Adam/clip_by_value_13/MinimumMinimumtraining/Adam/add_38training/Adam/Const_27*&
_output_shapes
:@*
T0

training/Adam/clip_by_value_13Maximum&training/Adam/clip_by_value_13/Minimumtraining/Adam/Const_26*&
_output_shapes
:@*
T0
n
training/Adam/Sqrt_13Sqrttraining/Adam/clip_by_value_13*
T0*&
_output_shapes
:@
[
training/Adam/add_39/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
{
training/Adam/add_39Addtraining/Adam/Sqrt_13training/Adam/add_39/y*
T0*&
_output_shapes
:@

training/Adam/truediv_13RealDivtraining/Adam/mul_65training/Adam/add_39*
T0*&
_output_shapes
:@
|
training/Adam/sub_40Subconv2d_6/kernel/readtraining/Adam/truediv_13*
T0*&
_output_shapes
:@
Ú
training/Adam/Assign_36Assigntraining/Adam/Variable_12training/Adam/add_37*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
Ú
training/Adam/Assign_37Assigntraining/Adam/Variable_26training/Adam/add_38*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_26*
validate_shape(*&
_output_shapes
:@
Ć
training/Adam/Assign_38Assignconv2d_6/kerneltraining/Adam/sub_40*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel*
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
training/Adam/sub_41Subtraining/Adam/sub_41/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_67Multraining/Adam/sub_419training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
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
training/Adam/Square_13Square9training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
o
training/Adam/mul_69Multraining/Adam/sub_42training/Adam/Square_13*
T0*
_output_shapes
:
l
training/Adam/add_41Addtraining/Adam/mul_68training/Adam/mul_69*
_output_shapes
:*
T0
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
training/Adam/clip_by_value_14Maximum&training/Adam/clip_by_value_14/Minimumtraining/Adam/Const_28*
_output_shapes
:*
T0
b
training/Adam/Sqrt_14Sqrttraining/Adam/clip_by_value_14*
_output_shapes
:*
T0
[
training/Adam/add_42/yConst*
valueB
 *żÖ3*
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
training/Adam/sub_43Subconv2d_6/bias/readtraining/Adam/truediv_14*
T0*
_output_shapes
:
Î
training/Adam/Assign_39Assigntraining/Adam/Variable_13training/Adam/add_40*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Î
training/Adam/Assign_40Assigntraining/Adam/Variable_27training/Adam/add_41*,
_class"
 loc:@training/Adam/Variable_27*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
ś
training/Adam/Assign_41Assignconv2d_6/biastraining/Adam/sub_43*
use_locking(*
T0* 
_class
loc:@conv2d_6/bias*
validate_shape(*
_output_shapes
:
ř
training/group_depsNoOp	^loss/mul^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_24^training/Adam/Assign_25^training/Adam/Assign_26^training/Adam/Assign_27^training/Adam/Assign_28^training/Adam/Assign_29^training/Adam/Assign_3^training/Adam/Assign_30^training/Adam/Assign_31^training/Adam/Assign_32^training/Adam/Assign_33^training/Adam/Assign_34^training/Adam/Assign_35^training/Adam/Assign_36^training/Adam/Assign_37^training/Adam/Assign_38^training/Adam/Assign_39^training/Adam/Assign_4^training/Adam/Assign_40^training/Adam/Assign_41^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9


group_depsNoOp	^loss/mul
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
IsVariableInitialized_2IsVariableInitializedconv2d_1/kernel*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
dtype0

IsVariableInitialized_3IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializedconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializedconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_6IsVariableInitializedconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_7IsVariableInitializedconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializedconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_9IsVariableInitializedconv2d_4/bias* 
_class
loc:@conv2d_4/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitializedconv2d_5/kernel*"
_class
loc:@conv2d_5/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_11IsVariableInitializedconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_12IsVariableInitializedconv2d_6/kernel*"
_class
loc:@conv2d_6/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_13IsVariableInitializedconv2d_6/bias* 
_class
loc:@conv2d_6/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_14IsVariableInitializedAdam/iterations*
_output_shapes
: *"
_class
loc:@Adam/iterations*
dtype0	
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
IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_8*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_8*
dtype0

IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0*
_output_shapes
: 

IsVariableInitialized_29IsVariableInitializedtraining/Adam/Variable_10*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_10
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
IsVariableInitialized_32IsVariableInitializedtraining/Adam/Variable_13*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_13

IsVariableInitialized_33IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes
: 

IsVariableInitialized_34IsVariableInitializedtraining/Adam/Variable_15*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_15

IsVariableInitialized_35IsVariableInitializedtraining/Adam/Variable_16*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_16*
dtype0
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
IsVariableInitialized_38IsVariableInitializedtraining/Adam/Variable_19*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_19*
dtype0

IsVariableInitialized_39IsVariableInitializedtraining/Adam/Variable_20*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_20

IsVariableInitialized_40IsVariableInitializedtraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
: 

IsVariableInitialized_41IsVariableInitializedtraining/Adam/Variable_22*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_22*
dtype0

IsVariableInitialized_42IsVariableInitializedtraining/Adam/Variable_23*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_23

IsVariableInitialized_43IsVariableInitializedtraining/Adam/Variable_24*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_24*
dtype0
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
IsVariableInitialized_47IsVariableInitializedtraining/Adam/Variable_28*,
_class"
 loc:@training/Adam/Variable_28*
dtype0*
_output_shapes
: 

IsVariableInitialized_48IsVariableInitializedtraining/Adam/Variable_29*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_29
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
IsVariableInitialized_53IsVariableInitializedtraining/Adam/Variable_34*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_34*
dtype0

IsVariableInitialized_54IsVariableInitializedtraining/Adam/Variable_35*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_35

IsVariableInitialized_55IsVariableInitializedtraining/Adam/Variable_36*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_36
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
IsVariableInitialized_58IsVariableInitializedtraining/Adam/Variable_39*,
_class"
 loc:@training/Adam/Variable_39*
dtype0*
_output_shapes
: 

IsVariableInitialized_59IsVariableInitializedtraining/Adam/Variable_40*,
_class"
 loc:@training/Adam/Variable_40*
dtype0*
_output_shapes
: 

IsVariableInitialized_60IsVariableInitializedtraining/Adam/Variable_41*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_41*
dtype0
đ
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^conv2d_3/bias/Assign^conv2d_3/kernel/Assign^conv2d_4/bias/Assign^conv2d_4/kernel/Assign^conv2d_5/bias/Assign^conv2d_5/kernel/Assign^conv2d_6/bias/Assign^conv2d_6/kernel/Assign^input/bias/Assign^input/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign!^training/Adam/Variable_24/Assign!^training/Adam/Variable_25/Assign!^training/Adam/Variable_26/Assign!^training/Adam/Variable_27/Assign!^training/Adam/Variable_28/Assign!^training/Adam/Variable_29/Assign ^training/Adam/Variable_3/Assign!^training/Adam/Variable_30/Assign!^training/Adam/Variable_31/Assign!^training/Adam/Variable_32/Assign!^training/Adam/Variable_33/Assign!^training/Adam/Variable_34/Assign!^training/Adam/Variable_35/Assign!^training/Adam/Variable_36/Assign!^training/Adam/Variable_37/Assign!^training/Adam/Variable_38/Assign!^training/Adam/Variable_39/Assign ^training/Adam/Variable_4/Assign!^training/Adam/Variable_40/Assign!^training/Adam/Variable_41/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign""Ć6
	variables¸6ľ6
T
input/kernel:0input/kernel/Assigninput/kernel/read:02input/random_uniform:08
E
input/bias:0input/bias/Assigninput/bias/read:02input/Const:08
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
training/Adam/Variable_41:0 training/Adam/Variable_41/Assign training/Adam/Variable_41/read:02training/Adam/zeros_41:08"Đ6
trainable_variables¸6ľ6
T
input/kernel:0input/kernel/Assigninput/kernel/read:02input/random_uniform:08
E
input/bias:0input/bias/Assigninput/bias/read:02input/Const:08
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
training/Adam/Variable_41:0 training/Adam/Variable_41/Assign training/Adam/Variable_41/read:02training/Adam/zeros_41:08Ĺ~ă_       ŁK"	łvJ5×A*

loss=Ał Ş       ČÁ	vJ5×A*

val_lossŢ=t,č       Ř-	âL5×A*

lossP\=ůa8       ŮÜ2	>L5×A*

val_lossL+ó>fôÓY       Ř-	§ę¸M5×A*

loss/=CˇV       ŮÜ2	Rí¸M5×A*

val_lossäďÔ>ľí3       Ř-	Ň˙bO5×A*

lossq2=A¤ŞŞ       ŮÜ2	ë cO5×A*

val_lossĐ0>ůw˙       Ř-	ÎŚQ5×A*

lossB!=Ů='       ŮÜ2	Î§Q5×A*

val_lossÉ	>dDN       Ř-	É°R5×A*

loss=?é.       ŮÜ2	Ë°R5×A*

val_loss)\>KřSw       Ř-	ĐT5×A*

loss=.Đ2Ç       ŮÜ2	żŃT5×A*

val_lossŘÝ>jć       Ř-	ň%NV5×A*

loss=ĄÄ       ŮÜ2	'NV5×A*

val_lossQl>ťÂ|l       Ř-	óöW5×A*

lossń=ó5b       ŮÜ2	öW5×A*

val_lossŽĹ>)Ąu       Ř-	ďţY5×A	*

lossŢ=Łv       ŮÜ2	  Y5×A	*

val_loss.P>Ĺ	p       Ř-	´ŞL[5×A
*

lossŘ=úgya       ŮÜ2	÷ŤL[5×A
*

val_loss\ý>8>2"       Ř-	ŔAő\5×A*

lossB=XŁ&       ŮÜ2	Bő\5×A*

val_lossó˘˝=ćM\       Ř-	ŘŁ^5×A*

lossA=;Ř>       ŮÜ2	żŁ^5×A*

val_lossú{P="sű       Ř-	ť{K`5×A*

lossk=ůŔ       ŮÜ2	ł|K`5×A*

val_loss'âĺ>Ü¤=       Ř-	o)ńa5×A*

lossÓ=i       ŮÜ2	0,ńa5×A*

val_loss­U=0öQ˙       Ř-	Tc5×A*

losst=uVŰ       ŮÜ2	O c5×A*

val_loss=4üÇő       Ř-	ĆEe5×A*

loss =A¨ü˝       ŮÜ2	Ee5×A*

val_lossô¸.=[Šîâ       Ř-	'ůéf5×A*

lossŻ˙=.C       ŮÜ2	@úéf5×A*

val_lossď˙ĺ>đÁ         Ř-	qťh5×A*

loss@=$gĘ       ŮÜ2	źťh5×A*

val_lossÎ=	B­3       Ř-	Ć|j5×A*

loss =ćsËc       ŮÜ2	2Ç|j5×A*

val_loss8¸ő=č]řp       Ř-	Ö2l5×A*

loss` =I       ŮÜ2	ä×2l5×A*

val_loss˘:ßďw       Ř-	!Ům5×A*

lossü=abÎ       ŮÜ2	÷Ům5×A*

val_lossYű>Ţ? S       Ř-	^o5×A*

lossçţ=˛\J       ŮÜ2	o5×A*

val_loss2[Đ>źż       Ř-	N,q5×A*

lossü=$4>       ŮÜ2	ôP,q5×A*

val_loss22G>ß*ě+       Ř-	IŮÎr5×A*

loss5ű=F       ŮÜ2	UÜÎr5×A*

val_lossÎ˘<yh/       Ř-	Ď{t5×A*

loss'ü=÷Żjh       ŮÜ2	WĐ{t5×A*

val_losszľÖ>˝       Ř-	őI&v5×A*

losstú=$É       ŮÜ2	˛L&v5×A*

val_loss!s>Nr4       Ř-	˝ßÎw5×A*

loss*ú=ÓŰ       ŮÜ2	÷äÎw5×A*

val_loss#ý=ĆŘ0       Ř-	Ą-y5×A*

loss­ř=KI       ŮÜ2	@0y5×A*

val_loss\=       Ř-	Gr{5×A*

loss¸ű=(0       ŮÜ2	Ir{5×A*

val_lossŇ=Ńp       Ř-	ÎUX}5×A*

loss˛ř=ďAW       ŮÜ2	uXX}5×A*

val_lossú,C>Ě<Ýä       Ř-	JE5×A*

loss:ů=Ôĺ       ŮÜ2	tF5×A*

val_lossEr+> Ch       Ř-	ł}š5×A *

lossâů= Ľě       ŮÜ2	Ž~š5×A *

val_loss\V=ţ>h       Ř-	d5×A!*

loss.ř=Ú˘       ŮÜ2	źd5×A!*

val_loss¨ęH=×ôK       Ř-	˘c5×A"*

loss0÷={*ü       ŮÜ2	d5×A"*

val_losszí×>÷\       Ř-	Đ¸5×A#*

lossP÷=Ú]       ŮÜ2	×¸5×A#*

val_lossBI;>47´       Ř-	Ta5×A$*

lossů=+lŽ       ŮÜ2	Ua5×A$*

val_lossŕÄ=ěR(J       Ř-	 5×A%*

lossľ÷=úĽw9       ŮÜ2	°5×A%*

val_lossž>ŘŘŻ.       Ř-	ÁÄˇ5×A&*

loss?ö=öQP1       ŮÜ2	eÇˇ5×A&*

val_loss.Š>ł;r!       Ř-	N`5×A'*

lossö=Ů¸°       ŮÜ2	`5×A'*

val_loss& >Š       Ř-	Ţ95×A(*

loss+ô=ŽĽ       ŮÜ2	ď:5×A(*

val_lossĘĄ=Á]č       Ř-	hź5×A)*

losslö=ÁĽ˝&       ŮÜ2	§ź5×A)*

val_lossÎ>uVH°       Ř-	?;5×A**

loss|ô=źţ4Š       ŮÜ2	=5×A**

val_loss°§2=ţYË       Ř-	BĎM5×A+*

loss!ô=śł|       ŮÜ2	 ĐM5×A+*

val_loss×@>FŔ¨       Ř-	íó5×A,*

lossűó=ćNÇ       ŮÜ2	îó5×A,*

val_loss2Ë(>]ś       Ř-	5×A-*

lossó=ß/#       ŮÜ2	Í5×A-*

val_lossb,Í>ŠX[       Ř-	N+C5×A.*

loss?ô=ż!×       ŮÜ2	.C5×A.*

val_loss}f<îÜĘ
       Ř-	.ë5×A/*

lossĂô=ě÷/       ŮÜ2	>ë5×A/*

val_lossÝ:Ĺ=k]9]       Ř-	 5×A0*

loss÷ô=´&.ň       ŮÜ2	>!5×A0*

val_loss7>ţaoÎ       Ř-	nh75×A1*

lossĹó=üűGŇ       ŮÜ2	+k75×A1*

val_lossT>[ ę       Ř-	Đä5×A2*

lossěń=ërP       ŮÜ2	öĐä5×A2*

val_lossů<=/ť       Ř-	Żî 5×A3*

lossZń=ôęąR       ŮÜ2	áď 5×A3*

val_lossç7D>ĆŃĂ       Ř-	č˝9˘5×A4*

lossľó=mŢfđ       ŮÜ2	Ŕ9˘5×A4*

val_lossÂ=13<       Ř-	iŤćŁ5×A5*

loss)ń=Î ž       ŮÜ2	ŹćŁ5×A5*

val_loss/řĄ=ž°ĚĆ       Ř-	ZeČĽ5×A6*

lossDö=V×iq       ŮÜ2	{fČĽ5×A6*

val_lossňí<pĹa       Ř-	Ľ{§5×A7*

lossŻń=˘ôĆN       ŮÜ2	§{§5×A7*

val_loss=ŹP>­;!       Ř-	2W$Š5×A8*

lossPń=,Úô       ŮÜ2	§Y$Š5×A8*

val_loss0V°>ĽťĆ       Ř-	aŃŞ5×A9*

lossÝđ=HvŞÓ       ŮÜ2	ŃŞ5×A9*

val_lossŘ7?Ďřl