       ŁK"	  tV$×Abrain.Event:2Ţč^2.     ÖY	0źtV$×A"ĽÜ
~
input_1Placeholder*
dtype0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*&
shape:˙˙˙˙˙˙˙˙˙¸Đ
v
conv1a/truncated_normal/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
conv1a/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv1a/truncated_normal/stddevConst*
valueB
 *k>*
dtype0*
_output_shapes
: 
ľ
'conv1a/truncated_normal/TruncatedNormalTruncatedNormalconv1a/truncated_normal/shape*
seedą˙ĺ)*
T0*
dtype0*&
_output_shapes
:@*
seed2Ş

conv1a/truncated_normal/mulMul'conv1a/truncated_normal/TruncatedNormalconv1a/truncated_normal/stddev*
T0*&
_output_shapes
:@

conv1a/truncated_normalAddconv1a/truncated_normal/mulconv1a/truncated_normal/mean*
T0*&
_output_shapes
:@

conv1a/kernel
VariableV2*
dtype0*&
_output_shapes
:@*
	container *
shape:@*
shared_name 
Â
conv1a/kernel/AssignAssignconv1a/kernelconv1a/truncated_normal*
use_locking(*
T0* 
_class
loc:@conv1a/kernel*
validate_shape(*&
_output_shapes
:@

conv1a/kernel/readIdentityconv1a/kernel*
T0* 
_class
loc:@conv1a/kernel*&
_output_shapes
:@
Y
conv1a/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
w
conv1a/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
Ľ
conv1a/bias/AssignAssignconv1a/biasconv1a/Const*
use_locking(*
T0*
_class
loc:@conv1a/bias*
validate_shape(*
_output_shapes
:@
n
conv1a/bias/readIdentityconv1a/bias*
T0*
_class
loc:@conv1a/bias*
_output_shapes
:@
q
 conv1a/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ă
conv1a/convolutionConv2Dinput_1conv1a/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
	dilations


conv1a/BiasAddBiasAddconv1a/convolutionconv1a/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
_
conv1a/ReluReluconv1a/BiasAdd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
v
conv1b/truncated_normal/shapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
a
conv1b/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
c
conv1b/truncated_normal/stddevConst*
valueB
 *¸1=*
dtype0*
_output_shapes
: 
ś
'conv1b/truncated_normal/TruncatedNormalTruncatedNormalconv1b/truncated_normal/shape*
seedą˙ĺ)*
T0*
dtype0*&
_output_shapes
:@@*
seed2˛ś

conv1b/truncated_normal/mulMul'conv1b/truncated_normal/TruncatedNormalconv1b/truncated_normal/stddev*
T0*&
_output_shapes
:@@

conv1b/truncated_normalAddconv1b/truncated_normal/mulconv1b/truncated_normal/mean*
T0*&
_output_shapes
:@@

conv1b/kernel
VariableV2*
shape:@@*
shared_name *
dtype0*&
_output_shapes
:@@*
	container 
Â
conv1b/kernel/AssignAssignconv1b/kernelconv1b/truncated_normal*
use_locking(*
T0* 
_class
loc:@conv1b/kernel*
validate_shape(*&
_output_shapes
:@@

conv1b/kernel/readIdentityconv1b/kernel* 
_class
loc:@conv1b/kernel*&
_output_shapes
:@@*
T0
Y
conv1b/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
w
conv1b/bias
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
Ľ
conv1b/bias/AssignAssignconv1b/biasconv1b/Const*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv1b/bias*
validate_shape(
n
conv1b/bias/readIdentityconv1b/bias*
T0*
_class
loc:@conv1b/bias*
_output_shapes
:@
q
 conv1b/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ç
conv1b/convolutionConv2Dconv1a/Reluconv1b/kernel/read*
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

conv1b/BiasAddBiasAddconv1b/convolutionconv1b/bias/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0*
data_formatNHWC
_
conv1b/ReluReluconv1b/BiasAdd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
´
pool1/MaxPoolMaxPoolconv1b/Relu*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
v
conv2a/truncated_normal/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
a
conv2a/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv2a/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *¸1=
ˇ
'conv2a/truncated_normal/TruncatedNormalTruncatedNormalconv2a/truncated_normal/shape*
seedą˙ĺ)*
T0*
dtype0*'
_output_shapes
:@*
seed2żŞ

conv2a/truncated_normal/mulMul'conv2a/truncated_normal/TruncatedNormalconv2a/truncated_normal/stddev*
T0*'
_output_shapes
:@

conv2a/truncated_normalAddconv2a/truncated_normal/mulconv2a/truncated_normal/mean*'
_output_shapes
:@*
T0

conv2a/kernel
VariableV2*
shared_name *
dtype0*'
_output_shapes
:@*
	container *
shape:@
Ă
conv2a/kernel/AssignAssignconv2a/kernelconv2a/truncated_normal*
use_locking(*
T0* 
_class
loc:@conv2a/kernel*
validate_shape(*'
_output_shapes
:@

conv2a/kernel/readIdentityconv2a/kernel*
T0* 
_class
loc:@conv2a/kernel*'
_output_shapes
:@
[
conv2a/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
y
conv2a/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ś
conv2a/bias/AssignAssignconv2a/biasconv2a/Const*
use_locking(*
T0*
_class
loc:@conv2a/bias*
validate_shape(*
_output_shapes	
:
o
conv2a/bias/readIdentityconv2a/bias*
T0*
_class
loc:@conv2a/bias*
_output_shapes	
:
q
 conv2a/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ę
conv2a/convolutionConv2Dpool1/MaxPoolconv2a/kernel/read*
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

conv2a/BiasAddBiasAddconv2a/convolutionconv2a/bias/read*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0*
data_formatNHWC
`
conv2a/ReluReluconv2a/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
v
conv2b/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
a
conv2b/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv2b/truncated_normal/stddevConst*
valueB
 *B=*
dtype0*
_output_shapes
: 
¸
'conv2b/truncated_normal/TruncatedNormalTruncatedNormalconv2b/truncated_normal/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2éŮŔ

conv2b/truncated_normal/mulMul'conv2b/truncated_normal/TruncatedNormalconv2b/truncated_normal/stddev*(
_output_shapes
:*
T0

conv2b/truncated_normalAddconv2b/truncated_normal/mulconv2b/truncated_normal/mean*(
_output_shapes
:*
T0

conv2b/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ä
conv2b/kernel/AssignAssignconv2b/kernelconv2b/truncated_normal*
use_locking(*
T0* 
_class
loc:@conv2b/kernel*
validate_shape(*(
_output_shapes
:

conv2b/kernel/readIdentityconv2b/kernel*(
_output_shapes
:*
T0* 
_class
loc:@conv2b/kernel
[
conv2b/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
y
conv2b/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ś
conv2b/bias/AssignAssignconv2b/biasconv2b/Const*
use_locking(*
T0*
_class
loc:@conv2b/bias*
validate_shape(*
_output_shapes	
:
o
conv2b/bias/readIdentityconv2b/bias*
_output_shapes	
:*
T0*
_class
loc:@conv2b/bias
q
 conv2b/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
č
conv2b/convolutionConv2Dconv2a/Reluconv2b/kernel/read*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv2b/BiasAddBiasAddconv2b/convolutionconv2b/bias/read*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0
`
conv2b/ReluReluconv2b/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
ľ
pool2/MaxPoolMaxPoolconv2b/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0
v
conv3a/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
a
conv3a/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
c
conv3a/truncated_normal/stddevConst*
valueB
 *B=*
dtype0*
_output_shapes
: 
¸
'conv3a/truncated_normal/TruncatedNormalTruncatedNormalconv3a/truncated_normal/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2ŤíŻ

conv3a/truncated_normal/mulMul'conv3a/truncated_normal/TruncatedNormalconv3a/truncated_normal/stddev*
T0*(
_output_shapes
:

conv3a/truncated_normalAddconv3a/truncated_normal/mulconv3a/truncated_normal/mean*
T0*(
_output_shapes
:

conv3a/kernel
VariableV2*
shared_name *
dtype0*(
_output_shapes
:*
	container *
shape:
Ä
conv3a/kernel/AssignAssignconv3a/kernelconv3a/truncated_normal*
use_locking(*
T0* 
_class
loc:@conv3a/kernel*
validate_shape(*(
_output_shapes
:

conv3a/kernel/readIdentityconv3a/kernel*
T0* 
_class
loc:@conv3a/kernel*(
_output_shapes
:
[
conv3a/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
y
conv3a/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ś
conv3a/bias/AssignAssignconv3a/biasconv3a/Const*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv3a/bias
o
conv3a/bias/readIdentityconv3a/bias*
_output_shapes	
:*
T0*
_class
loc:@conv3a/bias
q
 conv3a/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ę
conv3a/convolutionConv2Dpool2/MaxPoolconv3a/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙

conv3a/BiasAddBiasAddconv3a/convolutionconv3a/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
`
conv3a/ReluReluconv3a/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
v
conv3b/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
a
conv3b/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv3b/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *¸1	=*
dtype0
¸
'conv3b/truncated_normal/TruncatedNormalTruncatedNormalconv3b/truncated_normal/shape*
T0*
dtype0*(
_output_shapes
:*
seed2Ůá*
seedą˙ĺ)

conv3b/truncated_normal/mulMul'conv3b/truncated_normal/TruncatedNormalconv3b/truncated_normal/stddev*
T0*(
_output_shapes
:

conv3b/truncated_normalAddconv3b/truncated_normal/mulconv3b/truncated_normal/mean*
T0*(
_output_shapes
:

conv3b/kernel
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
Ä
conv3b/kernel/AssignAssignconv3b/kernelconv3b/truncated_normal*(
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv3b/kernel*
validate_shape(

conv3b/kernel/readIdentityconv3b/kernel*
T0* 
_class
loc:@conv3b/kernel*(
_output_shapes
:
[
conv3b/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
y
conv3b/bias
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Ś
conv3b/bias/AssignAssignconv3b/biasconv3b/Const*
_class
loc:@conv3b/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
o
conv3b/bias/readIdentityconv3b/bias*
T0*
_class
loc:@conv3b/bias*
_output_shapes	
:
q
 conv3b/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
č
conv3b/convolutionConv2Dconv3a/Reluconv3b/kernel/read*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv3b/BiasAddBiasAddconv3b/convolutionconv3b/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
`
conv3b/ReluReluconv3b/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
ľ
pool3/MaxPoolMaxPoolconv3b/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
v
conv4a/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
a
conv4a/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv4a/truncated_normal/stddevConst*
valueB
 *¸1	=*
dtype0*
_output_shapes
: 
ˇ
'conv4a/truncated_normal/TruncatedNormalTruncatedNormalconv4a/truncated_normal/shape*
T0*
dtype0*(
_output_shapes
:*
seed2żŔX*
seedą˙ĺ)

conv4a/truncated_normal/mulMul'conv4a/truncated_normal/TruncatedNormalconv4a/truncated_normal/stddev*
T0*(
_output_shapes
:

conv4a/truncated_normalAddconv4a/truncated_normal/mulconv4a/truncated_normal/mean*(
_output_shapes
:*
T0

conv4a/kernel
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
Ä
conv4a/kernel/AssignAssignconv4a/kernelconv4a/truncated_normal* 
_class
loc:@conv4a/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0

conv4a/kernel/readIdentityconv4a/kernel*
T0* 
_class
loc:@conv4a/kernel*(
_output_shapes
:
[
conv4a/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
y
conv4a/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ś
conv4a/bias/AssignAssignconv4a/biasconv4a/Const*
use_locking(*
T0*
_class
loc:@conv4a/bias*
validate_shape(*
_output_shapes	
:
o
conv4a/bias/readIdentityconv4a/bias*
T0*
_class
loc:@conv4a/bias*
_output_shapes	
:
q
 conv4a/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ę
conv4a/convolutionConv2Dpool3/MaxPoolconv4a/kernel/read*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME

conv4a/BiasAddBiasAddconv4a/convolutionconv4a/bias/read*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0
`
conv4a/ReluReluconv4a/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
v
conv4b/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
a
conv4b/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv4b/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *Â<
¸
'conv4b/truncated_normal/TruncatedNormalTruncatedNormalconv4b/truncated_normal/shape*(
_output_shapes
:*
seed2ÜŠÇ*
seedą˙ĺ)*
T0*
dtype0

conv4b/truncated_normal/mulMul'conv4b/truncated_normal/TruncatedNormalconv4b/truncated_normal/stddev*(
_output_shapes
:*
T0

conv4b/truncated_normalAddconv4b/truncated_normal/mulconv4b/truncated_normal/mean*
T0*(
_output_shapes
:

conv4b/kernel
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
Ä
conv4b/kernel/AssignAssignconv4b/kernelconv4b/truncated_normal* 
_class
loc:@conv4b/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0

conv4b/kernel/readIdentityconv4b/kernel*(
_output_shapes
:*
T0* 
_class
loc:@conv4b/kernel
[
conv4b/ConstConst*
_output_shapes	
:*
valueB*    *
dtype0
y
conv4b/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ś
conv4b/bias/AssignAssignconv4b/biasconv4b/Const*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv4b/bias*
validate_shape(
o
conv4b/bias/readIdentityconv4b/bias*
T0*
_class
loc:@conv4b/bias*
_output_shapes	
:
q
 conv4b/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
č
conv4b/convolutionConv2Dconv4a/Reluconv4b/kernel/read*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv4b/BiasAddBiasAddconv4b/convolutionconv4b/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
`
conv4b/ReluReluconv4b/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
b
 drop4/keras_learning_phase/inputConst*
value	B
 Z *
dtype0
*
_output_shapes
: 

drop4/keras_learning_phasePlaceholderWithDefault drop4/keras_learning_phase/input*
shape: *
dtype0
*
_output_shapes
: 
v
drop4/cond/SwitchSwitchdrop4/keras_learning_phasedrop4/keras_learning_phase*
T0
*
_output_shapes
: : 
U
drop4/cond/switch_tIdentitydrop4/cond/Switch:1*
T0
*
_output_shapes
: 
S
drop4/cond/switch_fIdentitydrop4/cond/Switch*
T0
*
_output_shapes
: 
[
drop4/cond/pred_idIdentitydrop4/keras_learning_phase*
T0
*
_output_shapes
: 
k
drop4/cond/mul/yConst^drop4/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
}
drop4/cond/mulMuldrop4/cond/mul/Switch:1drop4/cond/mul/y*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
ť
drop4/cond/mul/SwitchSwitchconv4b/Reludrop4/cond/pred_id*
T0*
_class
loc:@conv4b/Relu*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙Ę:˙˙˙˙˙˙˙˙˙Ę
r
drop4/cond/dropout/rateConst^drop4/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
f
drop4/cond/dropout/ShapeShapedrop4/cond/mul*
T0*
out_type0*
_output_shapes
:
s
drop4/cond/dropout/sub/xConst^drop4/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
q
drop4/cond/dropout/subSubdrop4/cond/dropout/sub/xdrop4/cond/dropout/rate*
T0*
_output_shapes
: 

%drop4/cond/dropout/random_uniform/minConst^drop4/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

%drop4/cond/dropout/random_uniform/maxConst^drop4/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ă
/drop4/cond/dropout/random_uniform/RandomUniformRandomUniformdrop4/cond/dropout/Shape*
T0*
dtype0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
seed2ůă*
seedą˙ĺ)

%drop4/cond/dropout/random_uniform/subSub%drop4/cond/dropout/random_uniform/max%drop4/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Á
%drop4/cond/dropout/random_uniform/mulMul/drop4/cond/dropout/random_uniform/RandomUniform%drop4/cond/dropout/random_uniform/sub*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
ł
!drop4/cond/dropout/random_uniformAdd%drop4/cond/dropout/random_uniform/mul%drop4/cond/dropout/random_uniform/min*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0

drop4/cond/dropout/addAdddrop4/cond/dropout/sub!drop4/cond/dropout/random_uniform*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
v
drop4/cond/dropout/FloorFloordrop4/cond/dropout/add*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

drop4/cond/dropout/truedivRealDivdrop4/cond/muldrop4/cond/dropout/sub*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

drop4/cond/dropout/mulMuldrop4/cond/dropout/truedivdrop4/cond/dropout/Floor*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0
š
drop4/cond/Switch_1Switchconv4b/Reludrop4/cond/pred_id*
T0*
_class
loc:@conv4b/Relu*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙Ę:˙˙˙˙˙˙˙˙˙Ę

drop4/cond/MergeMergedrop4/cond/Switch_1drop4/cond/dropout/mul*
N*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙Ę: *
T0
¸
pool4/MaxPoolMaxPooldrop4/cond/Merge*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
v
conv5a/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
a
conv5a/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv5a/truncated_normal/stddevConst*
valueB
 *Â<*
dtype0*
_output_shapes
: 
¸
'conv5a/truncated_normal/TruncatedNormalTruncatedNormalconv5a/truncated_normal/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2Šż

conv5a/truncated_normal/mulMul'conv5a/truncated_normal/TruncatedNormalconv5a/truncated_normal/stddev*(
_output_shapes
:*
T0

conv5a/truncated_normalAddconv5a/truncated_normal/mulconv5a/truncated_normal/mean*(
_output_shapes
:*
T0

conv5a/kernel
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
Ä
conv5a/kernel/AssignAssignconv5a/kernelconv5a/truncated_normal*
use_locking(*
T0* 
_class
loc:@conv5a/kernel*
validate_shape(*(
_output_shapes
:

conv5a/kernel/readIdentityconv5a/kernel*(
_output_shapes
:*
T0* 
_class
loc:@conv5a/kernel
[
conv5a/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
y
conv5a/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ś
conv5a/bias/AssignAssignconv5a/biasconv5a/Const*
use_locking(*
T0*
_class
loc:@conv5a/bias*
validate_shape(*
_output_shapes	
:
o
conv5a/bias/readIdentityconv5a/bias*
_output_shapes	
:*
T0*
_class
loc:@conv5a/bias
q
 conv5a/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
č
conv5a/convolutionConv2Dpool4/MaxPoolconv5a/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

conv5a/BiasAddBiasAddconv5a/convolutionconv5a/bias/read*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
T0
^
conv5a/ReluReluconv5a/BiasAdd*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
v
conv5b/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
a
conv5b/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv5b/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *¸1<
¸
'conv5b/truncated_normal/TruncatedNormalTruncatedNormalconv5b/truncated_normal/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2üÍ

conv5b/truncated_normal/mulMul'conv5b/truncated_normal/TruncatedNormalconv5b/truncated_normal/stddev*
T0*(
_output_shapes
:

conv5b/truncated_normalAddconv5b/truncated_normal/mulconv5b/truncated_normal/mean*
T0*(
_output_shapes
:

conv5b/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ä
conv5b/kernel/AssignAssignconv5b/kernelconv5b/truncated_normal*
use_locking(*
T0* 
_class
loc:@conv5b/kernel*
validate_shape(*(
_output_shapes
:

conv5b/kernel/readIdentityconv5b/kernel*(
_output_shapes
:*
T0* 
_class
loc:@conv5b/kernel
[
conv5b/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
y
conv5b/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
shared_name *
dtype0
Ś
conv5b/bias/AssignAssignconv5b/biasconv5b/Const*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv5b/bias*
validate_shape(
o
conv5b/bias/readIdentityconv5b/bias*
T0*
_class
loc:@conv5b/bias*
_output_shapes	
:
q
 conv5b/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ć
conv5b/convolutionConv2Dconv5a/Reluconv5b/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
	dilations
*
T0

conv5b/BiasAddBiasAddconv5b/convolutionconv5b/bias/read*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
T0
^
conv5b/ReluReluconv5b/BiasAdd*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
v
drop5/cond/SwitchSwitchdrop4/keras_learning_phasedrop4/keras_learning_phase*
_output_shapes
: : *
T0

U
drop5/cond/switch_tIdentitydrop5/cond/Switch:1*
_output_shapes
: *
T0

S
drop5/cond/switch_fIdentitydrop5/cond/Switch*
_output_shapes
: *
T0

[
drop5/cond/pred_idIdentitydrop4/keras_learning_phase*
T0
*
_output_shapes
: 
k
drop5/cond/mul/yConst^drop5/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
{
drop5/cond/mulMuldrop5/cond/mul/Switch:1drop5/cond/mul/y*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
ˇ
drop5/cond/mul/SwitchSwitchconv5b/Reludrop5/cond/pred_id*
_class
loc:@conv5b/Relu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙Ce:˙˙˙˙˙˙˙˙˙Ce*
T0
r
drop5/cond/dropout/rateConst^drop5/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
f
drop5/cond/dropout/ShapeShapedrop5/cond/mul*
T0*
out_type0*
_output_shapes
:
s
drop5/cond/dropout/sub/xConst^drop5/cond/switch_t*
_output_shapes
: *
valueB
 *  ?*
dtype0
q
drop5/cond/dropout/subSubdrop5/cond/dropout/sub/xdrop5/cond/dropout/rate*
T0*
_output_shapes
: 

%drop5/cond/dropout/random_uniform/minConst^drop5/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

%drop5/cond/dropout/random_uniform/maxConst^drop5/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Á
/drop5/cond/dropout/random_uniform/RandomUniformRandomUniformdrop5/cond/dropout/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
seed2ŁĎ*
seedą˙ĺ)*
T0*
dtype0

%drop5/cond/dropout/random_uniform/subSub%drop5/cond/dropout/random_uniform/max%drop5/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
ż
%drop5/cond/dropout/random_uniform/mulMul/drop5/cond/dropout/random_uniform/RandomUniform%drop5/cond/dropout/random_uniform/sub*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
T0
ą
!drop5/cond/dropout/random_uniformAdd%drop5/cond/dropout/random_uniform/mul%drop5/cond/dropout/random_uniform/min*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

drop5/cond/dropout/addAdddrop5/cond/dropout/sub!drop5/cond/dropout/random_uniform*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
T0
t
drop5/cond/dropout/FloorFloordrop5/cond/dropout/add*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

drop5/cond/dropout/truedivRealDivdrop5/cond/muldrop5/cond/dropout/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

drop5/cond/dropout/mulMuldrop5/cond/dropout/truedivdrop5/cond/dropout/Floor*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
ľ
drop5/cond/Switch_1Switchconv5b/Reludrop5/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙Ce:˙˙˙˙˙˙˙˙˙Ce*
T0*
_class
loc:@conv5b/Relu

drop5/cond/MergeMergedrop5/cond/Switch_1drop5/cond/dropout/mul*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ce: 
Y
	up1/ShapeShapedrop5/cond/Merge*
T0*
out_type0*
_output_shapes
:
a
up1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
c
up1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
c
up1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

up1/strided_sliceStridedSlice	up1/Shapeup1/strided_slice/stackup1/strided_slice/stack_1up1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
Z
	up1/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
Q
up1/mulMulup1/strided_slice	up1/Const*
_output_shapes
:*
T0

up1/ResizeNearestNeighborResizeNearestNeighbordrop5/cond/Mergeup1/mul*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
align_corners( 
u
conv6/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
`
conv6/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
b
conv6/truncated_normal/stddevConst*
valueB
 *ĘÍ<*
dtype0*
_output_shapes
: 
ś
&conv6/truncated_normal/TruncatedNormalTruncatedNormalconv6/truncated_normal/shape*
dtype0*(
_output_shapes
:*
seed2Łż*
seedą˙ĺ)*
T0

conv6/truncated_normal/mulMul&conv6/truncated_normal/TruncatedNormalconv6/truncated_normal/stddev*
T0*(
_output_shapes
:

conv6/truncated_normalAddconv6/truncated_normal/mulconv6/truncated_normal/mean*(
_output_shapes
:*
T0

conv6/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ŕ
conv6/kernel/AssignAssignconv6/kernelconv6/truncated_normal*
_class
loc:@conv6/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0

conv6/kernel/readIdentityconv6/kernel*(
_output_shapes
:*
T0*
_class
loc:@conv6/kernel
Z
conv6/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
x

conv6/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
˘
conv6/bias/AssignAssign
conv6/biasconv6/Const*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv6/bias
l
conv6/bias/readIdentity
conv6/bias*
_output_shapes	
:*
T0*
_class
loc:@conv6/bias
p
conv6/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
ô
conv6/convolutionConv2Dup1/ResizeNearestNeighborconv6/kernel/read*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME

conv6/BiasAddBiasAddconv6/convolutionconv6/bias/read*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0
^

conv6/ReluReluconv6/BiasAdd*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0

zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*9
value0B."                                *
dtype0

zero_padding2d_1/PadPad
conv6/Reluzero_padding2d_1/Pad/paddings*
T0*
	Tpaddings0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
[
concatenate_1/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
ľ
concatenate_1/concatConcatV2drop4/cond/Mergezero_padding2d_1/Padconcatenate_1/concat/axis*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*

Tidx0
v
conv7a/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
a
conv7a/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv7a/truncated_normal/stddevConst*
valueB
 *¸1<*
dtype0*
_output_shapes
: 
¸
'conv7a/truncated_normal/TruncatedNormalTruncatedNormalconv7a/truncated_normal/shape*
T0*
dtype0*(
_output_shapes
:*
seed2Đůç*
seedą˙ĺ)

conv7a/truncated_normal/mulMul'conv7a/truncated_normal/TruncatedNormalconv7a/truncated_normal/stddev*(
_output_shapes
:*
T0

conv7a/truncated_normalAddconv7a/truncated_normal/mulconv7a/truncated_normal/mean*(
_output_shapes
:*
T0

conv7a/kernel
VariableV2*
shared_name *
dtype0*(
_output_shapes
:*
	container *
shape:
Ä
conv7a/kernel/AssignAssignconv7a/kernelconv7a/truncated_normal*
use_locking(*
T0* 
_class
loc:@conv7a/kernel*
validate_shape(*(
_output_shapes
:

conv7a/kernel/readIdentityconv7a/kernel*(
_output_shapes
:*
T0* 
_class
loc:@conv7a/kernel
[
conv7a/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
y
conv7a/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ś
conv7a/bias/AssignAssignconv7a/biasconv7a/Const*
T0*
_class
loc:@conv7a/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
o
conv7a/bias/readIdentityconv7a/bias*
T0*
_class
loc:@conv7a/bias*
_output_shapes	
:
q
 conv7a/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ń
conv7a/convolutionConv2Dconcatenate_1/concatconv7a/kernel/read*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations
*
T0*
strides
*
data_formatNHWC

conv7a/BiasAddBiasAddconv7a/convolutionconv7a/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
`
conv7a/ReluReluconv7a/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
v
conv7b/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
a
conv7b/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv7b/truncated_normal/stddevConst*
valueB
 *Â<*
dtype0*
_output_shapes
: 
¸
'conv7b/truncated_normal/TruncatedNormalTruncatedNormalconv7b/truncated_normal/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2ÍĄţ

conv7b/truncated_normal/mulMul'conv7b/truncated_normal/TruncatedNormalconv7b/truncated_normal/stddev*(
_output_shapes
:*
T0

conv7b/truncated_normalAddconv7b/truncated_normal/mulconv7b/truncated_normal/mean*
T0*(
_output_shapes
:

conv7b/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ä
conv7b/kernel/AssignAssignconv7b/kernelconv7b/truncated_normal*
use_locking(*
T0* 
_class
loc:@conv7b/kernel*
validate_shape(*(
_output_shapes
:

conv7b/kernel/readIdentityconv7b/kernel*(
_output_shapes
:*
T0* 
_class
loc:@conv7b/kernel
[
conv7b/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
y
conv7b/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ś
conv7b/bias/AssignAssignconv7b/biasconv7b/Const*
use_locking(*
T0*
_class
loc:@conv7b/bias*
validate_shape(*
_output_shapes	
:
o
conv7b/bias/readIdentityconv7b/bias*
T0*
_class
loc:@conv7b/bias*
_output_shapes	
:
q
 conv7b/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
č
conv7b/convolutionConv2Dconv7a/Reluconv7b/kernel/read*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv7b/BiasAddBiasAddconv7b/convolutionconv7b/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
`
conv7b/ReluReluconv7b/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
`
up_sampling2d_1/ShapeShapeconv7b/Relu*
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
Í
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape#up_sampling2d_1/strided_slice/stack%up_sampling2d_1/strided_slice/stack_1%up_sampling2d_1/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0*
shrink_axis_mask 
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
˛
%up_sampling2d_1/ResizeNearestNeighborResizeNearestNeighborconv7b/Reluup_sampling2d_1/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
align_corners( *
T0
x
conv2d_1/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
c
conv2d_1/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 conv2d_1/truncated_normal/stddevConst*
valueB
 *6=*
dtype0*
_output_shapes
: 
ź
)conv2d_1/truncated_normal/TruncatedNormalTruncatedNormalconv2d_1/truncated_normal/shape*
T0*
dtype0*(
_output_shapes
:*
seed2ŐĽ*
seedą˙ĺ)
¤
conv2d_1/truncated_normal/mulMul)conv2d_1/truncated_normal/TruncatedNormal conv2d_1/truncated_normal/stddev*
T0*(
_output_shapes
:

conv2d_1/truncated_normalAddconv2d_1/truncated_normal/mulconv2d_1/truncated_normal/mean*
T0*(
_output_shapes
:

conv2d_1/kernel
VariableV2*
shared_name *
dtype0*(
_output_shapes
:*
	container *
shape:
Ě
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/truncated_normal*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*(
_output_shapes
:

conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*(
_output_shapes
:*
T0
]
conv2d_1/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_1/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ž
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes	
:
u
conv2d_1/bias/readIdentityconv2d_1/bias*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes	
:
s
"conv2d_1/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0

conv2d_1/convolutionConv2D%up_sampling2d_1/ResizeNearestNeighborconv2d_1/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙

conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
d
conv2d_1/ReluReluconv2d_1/BiasAdd*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0
[
concatenate_2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Š
concatenate_2/concatConcatV2conv3b/Reluconv2d_1/Reluconcatenate_2/concat/axis*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0
x
conv2d_2/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
c
conv2d_2/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
e
 conv2d_2/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *Â<
ź
)conv2d_2/truncated_normal/TruncatedNormalTruncatedNormalconv2d_2/truncated_normal/shape*
dtype0*(
_output_shapes
:*
seed2ĄÍĺ*
seedą˙ĺ)*
T0
¤
conv2d_2/truncated_normal/mulMul)conv2d_2/truncated_normal/TruncatedNormal conv2d_2/truncated_normal/stddev*(
_output_shapes
:*
T0

conv2d_2/truncated_normalAddconv2d_2/truncated_normal/mulconv2d_2/truncated_normal/mean*
T0*(
_output_shapes
:

conv2d_2/kernel
VariableV2*
shared_name *
dtype0*(
_output_shapes
:*
	container *
shape:
Ě
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/truncated_normal*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*(
_output_shapes
:

conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*(
_output_shapes
:
]
conv2d_2/ConstConst*
_output_shapes	
:*
valueB*    *
dtype0
{
conv2d_2/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
shared_name *
dtype0
Ž
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes	
:
u
conv2d_2/bias/readIdentityconv2d_2/bias*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes	
:
s
"conv2d_2/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
ő
conv2d_2/convolutionConv2Dconcatenate_2/concatconv2d_2/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙

conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0
d
conv2d_2/ReluReluconv2d_2/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
x
conv2d_3/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
c
conv2d_3/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 conv2d_3/truncated_normal/stddevConst*
valueB
 *¸1	=*
dtype0*
_output_shapes
: 
ź
)conv2d_3/truncated_normal/TruncatedNormalTruncatedNormalconv2d_3/truncated_normal/shape*
T0*
dtype0*(
_output_shapes
:*
seed2ţŞĂ*
seedą˙ĺ)
¤
conv2d_3/truncated_normal/mulMul)conv2d_3/truncated_normal/TruncatedNormal conv2d_3/truncated_normal/stddev*
T0*(
_output_shapes
:

conv2d_3/truncated_normalAddconv2d_3/truncated_normal/mulconv2d_3/truncated_normal/mean*(
_output_shapes
:*
T0

conv2d_3/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ě
conv2d_3/kernel/AssignAssignconv2d_3/kernelconv2d_3/truncated_normal*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(*(
_output_shapes
:

conv2d_3/kernel/readIdentityconv2d_3/kernel*
T0*"
_class
loc:@conv2d_3/kernel*(
_output_shapes
:
]
conv2d_3/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_3/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ž
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/Const* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
u
conv2d_3/bias/readIdentityconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
_output_shapes	
:*
T0
s
"conv2d_3/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
î
conv2d_3/convolutionConv2Dconv2d_2/Reluconv2d_3/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙

conv2d_3/BiasAddBiasAddconv2d_3/convolutionconv2d_3/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
d
conv2d_3/ReluReluconv2d_3/BiasAdd*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0
b
up_sampling2d_2/ShapeShapeconv2d_3/Relu*
T0*
out_type0*
_output_shapes
:
m
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
o
%up_sampling2d_2/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
o
%up_sampling2d_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Í
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape#up_sampling2d_2/strided_slice/stack%up_sampling2d_2/strided_slice/stack_1%up_sampling2d_2/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask 
f
up_sampling2d_2/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
u
up_sampling2d_2/mulMulup_sampling2d_2/strided_sliceup_sampling2d_2/Const*
T0*
_output_shapes
:
´
%up_sampling2d_2/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Reluup_sampling2d_2/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
align_corners( *
T0
x
conv2d_4/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
c
conv2d_4/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 conv2d_4/truncated_normal/stddevConst*
valueB
 *ĘM=*
dtype0*
_output_shapes
: 
ź
)conv2d_4/truncated_normal/TruncatedNormalTruncatedNormalconv2d_4/truncated_normal/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2áé
¤
conv2d_4/truncated_normal/mulMul)conv2d_4/truncated_normal/TruncatedNormal conv2d_4/truncated_normal/stddev*(
_output_shapes
:*
T0

conv2d_4/truncated_normalAddconv2d_4/truncated_normal/mulconv2d_4/truncated_normal/mean*(
_output_shapes
:*
T0

conv2d_4/kernel
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
Ě
conv2d_4/kernel/AssignAssignconv2d_4/kernelconv2d_4/truncated_normal*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*(
_output_shapes
:

conv2d_4/kernel/readIdentityconv2d_4/kernel*(
_output_shapes
:*
T0*"
_class
loc:@conv2d_4/kernel
]
conv2d_4/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_4/bias
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Ž
conv2d_4/bias/AssignAssignconv2d_4/biasconv2d_4/Const* 
_class
loc:@conv2d_4/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
u
conv2d_4/bias/readIdentityconv2d_4/bias* 
_class
loc:@conv2d_4/bias*
_output_shapes	
:*
T0
s
"conv2d_4/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

conv2d_4/convolutionConv2D%up_sampling2d_2/ResizeNearestNeighborconv2d_4/kernel/read*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME

conv2d_4/BiasAddBiasAddconv2d_4/convolutionconv2d_4/bias/read*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0
d
conv2d_4/ReluReluconv2d_4/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
[
concatenate_3/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
Š
concatenate_3/concatConcatV2conv2b/Reluconv2d_4/Reluconcatenate_3/concat/axis*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*

Tidx0
x
conv2d_5/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
c
conv2d_5/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 conv2d_5/truncated_normal/stddevConst*
valueB
 *¸1	=*
dtype0*
_output_shapes
: 
ť
)conv2d_5/truncated_normal/TruncatedNormalTruncatedNormalconv2d_5/truncated_normal/shape*
seedą˙ĺ)*
T0*
dtype0*(
_output_shapes
:*
seed2â¨y
¤
conv2d_5/truncated_normal/mulMul)conv2d_5/truncated_normal/TruncatedNormal conv2d_5/truncated_normal/stddev*
T0*(
_output_shapes
:

conv2d_5/truncated_normalAddconv2d_5/truncated_normal/mulconv2d_5/truncated_normal/mean*
T0*(
_output_shapes
:

conv2d_5/kernel
VariableV2*(
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
Ě
conv2d_5/kernel/AssignAssignconv2d_5/kernelconv2d_5/truncated_normal*
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel*
validate_shape(*(
_output_shapes
:

conv2d_5/kernel/readIdentityconv2d_5/kernel*"
_class
loc:@conv2d_5/kernel*(
_output_shapes
:*
T0
]
conv2d_5/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_5/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ž
conv2d_5/bias/AssignAssignconv2d_5/biasconv2d_5/Const* 
_class
loc:@conv2d_5/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
u
conv2d_5/bias/readIdentityconv2d_5/bias*
_output_shapes	
:*
T0* 
_class
loc:@conv2d_5/bias
s
"conv2d_5/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ő
conv2d_5/convolutionConv2Dconcatenate_3/concatconv2d_5/kernel/read*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv2d_5/BiasAddBiasAddconv2d_5/convolutionconv2d_5/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
d
conv2d_5/ReluReluconv2d_5/BiasAdd*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0
x
conv2d_6/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
c
conv2d_6/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 conv2d_6/truncated_normal/stddevConst*
valueB
 *B=*
dtype0*
_output_shapes
: 
ź
)conv2d_6/truncated_normal/TruncatedNormalTruncatedNormalconv2d_6/truncated_normal/shape*
T0*
dtype0*(
_output_shapes
:*
seed2ŤŐ*
seedą˙ĺ)
¤
conv2d_6/truncated_normal/mulMul)conv2d_6/truncated_normal/TruncatedNormal conv2d_6/truncated_normal/stddev*
T0*(
_output_shapes
:

conv2d_6/truncated_normalAddconv2d_6/truncated_normal/mulconv2d_6/truncated_normal/mean*(
_output_shapes
:*
T0

conv2d_6/kernel
VariableV2*
shared_name *
dtype0*(
_output_shapes
:*
	container *
shape:
Ě
conv2d_6/kernel/AssignAssignconv2d_6/kernelconv2d_6/truncated_normal*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*(
_output_shapes
:

conv2d_6/kernel/readIdentityconv2d_6/kernel*
T0*"
_class
loc:@conv2d_6/kernel*(
_output_shapes
:
]
conv2d_6/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_6/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ž
conv2d_6/bias/AssignAssignconv2d_6/biasconv2d_6/Const*
use_locking(*
T0* 
_class
loc:@conv2d_6/bias*
validate_shape(*
_output_shapes	
:
u
conv2d_6/bias/readIdentityconv2d_6/bias*
T0* 
_class
loc:@conv2d_6/bias*
_output_shapes	
:
s
"conv2d_6/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
î
conv2d_6/convolutionConv2Dconv2d_5/Reluconv2d_6/kernel/read*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv2d_6/BiasAddBiasAddconv2d_6/convolutionconv2d_6/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
d
conv2d_6/ReluReluconv2d_6/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
b
up_sampling2d_3/ShapeShapeconv2d_6/Relu*
_output_shapes
:*
T0*
out_type0
m
#up_sampling2d_3/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Í
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape#up_sampling2d_3/strided_slice/stack%up_sampling2d_3/strided_slice/stack_1%up_sampling2d_3/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:
f
up_sampling2d_3/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
u
up_sampling2d_3/mulMulup_sampling2d_3/strided_sliceup_sampling2d_3/Const*
T0*
_output_shapes
:
´
%up_sampling2d_3/ResizeNearestNeighborResizeNearestNeighborconv2d_6/Reluup_sampling2d_3/mul*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¸Đ*
align_corners( 
x
conv2d_7/truncated_normal/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
c
conv2d_7/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 conv2d_7/truncated_normal/stddevConst*
valueB
 *6=*
dtype0*
_output_shapes
: 
ť
)conv2d_7/truncated_normal/TruncatedNormalTruncatedNormalconv2d_7/truncated_normal/shape*
T0*
dtype0*'
_output_shapes
:@*
seed2÷áĹ*
seedą˙ĺ)
Ł
conv2d_7/truncated_normal/mulMul)conv2d_7/truncated_normal/TruncatedNormal conv2d_7/truncated_normal/stddev*
T0*'
_output_shapes
:@

conv2d_7/truncated_normalAddconv2d_7/truncated_normal/mulconv2d_7/truncated_normal/mean*
T0*'
_output_shapes
:@

conv2d_7/kernel
VariableV2*
dtype0*'
_output_shapes
:@*
	container *
shape:@*
shared_name 
Ë
conv2d_7/kernel/AssignAssignconv2d_7/kernelconv2d_7/truncated_normal*
T0*"
_class
loc:@conv2d_7/kernel*
validate_shape(*'
_output_shapes
:@*
use_locking(

conv2d_7/kernel/readIdentityconv2d_7/kernel*
T0*"
_class
loc:@conv2d_7/kernel*'
_output_shapes
:@
[
conv2d_7/ConstConst*
_output_shapes
:@*
valueB@*    *
dtype0
y
conv2d_7/bias
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
­
conv2d_7/bias/AssignAssignconv2d_7/biasconv2d_7/Const*
use_locking(*
T0* 
_class
loc:@conv2d_7/bias*
validate_shape(*
_output_shapes
:@
t
conv2d_7/bias/readIdentityconv2d_7/bias*
_output_shapes
:@*
T0* 
_class
loc:@conv2d_7/bias
s
"conv2d_7/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

conv2d_7/convolutionConv2D%up_sampling2d_3/ResizeNearestNeighborconv2d_7/kernel/read*1
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

conv2d_7/BiasAddBiasAddconv2d_7/convolutionconv2d_7/bias/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0*
data_formatNHWC
c
conv2d_7/ReluReluconv2d_7/BiasAdd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
[
concatenate_4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Š
concatenate_4/concatConcatV2conv1b/Reluconv2d_7/Reluconcatenate_4/concat/axis*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¸Đ*

Tidx0
x
conv2d_8/truncated_normal/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
c
conv2d_8/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
e
 conv2d_8/truncated_normal/stddevConst*
valueB
 *B=*
dtype0*
_output_shapes
: 
ť
)conv2d_8/truncated_normal/TruncatedNormalTruncatedNormalconv2d_8/truncated_normal/shape*
dtype0*'
_output_shapes
:@*
seed2ËËĽ*
seedą˙ĺ)*
T0
Ł
conv2d_8/truncated_normal/mulMul)conv2d_8/truncated_normal/TruncatedNormal conv2d_8/truncated_normal/stddev*'
_output_shapes
:@*
T0

conv2d_8/truncated_normalAddconv2d_8/truncated_normal/mulconv2d_8/truncated_normal/mean*'
_output_shapes
:@*
T0

conv2d_8/kernel
VariableV2*
shared_name *
dtype0*'
_output_shapes
:@*
	container *
shape:@
Ë
conv2d_8/kernel/AssignAssignconv2d_8/kernelconv2d_8/truncated_normal*
use_locking(*
T0*"
_class
loc:@conv2d_8/kernel*
validate_shape(*'
_output_shapes
:@

conv2d_8/kernel/readIdentityconv2d_8/kernel*
T0*"
_class
loc:@conv2d_8/kernel*'
_output_shapes
:@
[
conv2d_8/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_8/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
­
conv2d_8/bias/AssignAssignconv2d_8/biasconv2d_8/Const*
use_locking(*
T0* 
_class
loc:@conv2d_8/bias*
validate_shape(*
_output_shapes
:@
t
conv2d_8/bias/readIdentityconv2d_8/bias*
T0* 
_class
loc:@conv2d_8/bias*
_output_shapes
:@
s
"conv2d_8/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ô
conv2d_8/convolutionConv2Dconcatenate_4/concatconv2d_8/kernel/read*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv2d_8/BiasAddBiasAddconv2d_8/convolutionconv2d_8/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
c
conv2d_8/ReluReluconv2d_8/BiasAdd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
x
conv2d_9/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
c
conv2d_9/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 conv2d_9/truncated_normal/stddevConst*
valueB
 *¸1=*
dtype0*
_output_shapes
: 
š
)conv2d_9/truncated_normal/TruncatedNormalTruncatedNormalconv2d_9/truncated_normal/shape*
seedą˙ĺ)*
T0*
dtype0*&
_output_shapes
:@@*
seed2Óť
˘
conv2d_9/truncated_normal/mulMul)conv2d_9/truncated_normal/TruncatedNormal conv2d_9/truncated_normal/stddev*
T0*&
_output_shapes
:@@

conv2d_9/truncated_normalAddconv2d_9/truncated_normal/mulconv2d_9/truncated_normal/mean*
T0*&
_output_shapes
:@@

conv2d_9/kernel
VariableV2*
dtype0*&
_output_shapes
:@@*
	container *
shape:@@*
shared_name 
Ę
conv2d_9/kernel/AssignAssignconv2d_9/kernelconv2d_9/truncated_normal*
use_locking(*
T0*"
_class
loc:@conv2d_9/kernel*
validate_shape(*&
_output_shapes
:@@

conv2d_9/kernel/readIdentityconv2d_9/kernel*
T0*"
_class
loc:@conv2d_9/kernel*&
_output_shapes
:@@
[
conv2d_9/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_9/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
­
conv2d_9/bias/AssignAssignconv2d_9/biasconv2d_9/Const*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@conv2d_9/bias*
validate_shape(
t
conv2d_9/bias/readIdentityconv2d_9/bias*
T0* 
_class
loc:@conv2d_9/bias*
_output_shapes
:@
s
"conv2d_9/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
í
conv2d_9/convolutionConv2Dconv2d_8/Reluconv2d_9/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@

conv2d_9/BiasAddBiasAddconv2d_9/convolutionconv2d_9/bias/read*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0
c
conv2d_9/ReluReluconv2d_9/BiasAdd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
w
conv2d_10/random_uniform/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
a
conv2d_10/random_uniform/minConst*
valueB
 *Ş7ž*
dtype0*
_output_shapes
: 
a
conv2d_10/random_uniform/maxConst*
valueB
 *Ş7>*
dtype0*
_output_shapes
: 
ł
&conv2d_10/random_uniform/RandomUniformRandomUniformconv2d_10/random_uniform/shape*
T0*
dtype0*&
_output_shapes
:@*
seed2É˝*
seedą˙ĺ)

conv2d_10/random_uniform/subSubconv2d_10/random_uniform/maxconv2d_10/random_uniform/min*
_output_shapes
: *
T0

conv2d_10/random_uniform/mulMul&conv2d_10/random_uniform/RandomUniformconv2d_10/random_uniform/sub*
T0*&
_output_shapes
:@

conv2d_10/random_uniformAddconv2d_10/random_uniform/mulconv2d_10/random_uniform/min*&
_output_shapes
:@*
T0

conv2d_10/kernel
VariableV2*
dtype0*&
_output_shapes
:@*
	container *
shape:@*
shared_name 
Ě
conv2d_10/kernel/AssignAssignconv2d_10/kernelconv2d_10/random_uniform*&
_output_shapes
:@*
use_locking(*
T0*#
_class
loc:@conv2d_10/kernel*
validate_shape(

conv2d_10/kernel/readIdentityconv2d_10/kernel*&
_output_shapes
:@*
T0*#
_class
loc:@conv2d_10/kernel
\
conv2d_10/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
z
conv2d_10/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ą
conv2d_10/bias/AssignAssignconv2d_10/biasconv2d_10/Const*
T0*!
_class
loc:@conv2d_10/bias*
validate_shape(*
_output_shapes
:*
use_locking(
w
conv2d_10/bias/readIdentityconv2d_10/bias*
T0*!
_class
loc:@conv2d_10/bias*
_output_shapes
:
t
#conv2d_10/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
đ
conv2d_10/convolutionConv2Dconv2d_9/Reluconv2d_10/kernel/read*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv2d_10/BiasAddBiasAddconv2d_10/convolutionconv2d_10/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
k
conv2d_10/SigmoidSigmoidconv2d_10/BiasAdd*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
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
dtype0	*
_output_shapes
: *
	container *
shape: *
shared_name 
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
 *ˇŃ8*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0

Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
dtype0*
_output_shapes
: *
	container *
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
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
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
Adam/beta_2/readIdentityAdam/beta_2*
_output_shapes
: *
T0*
_class
loc:@Adam/beta_2
]
Adam/decay/initial_valueConst*
valueB
 *ŹĹ'7*
dtype0*
_output_shapes
: 
n

Adam/decay
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ş
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/decay*
validate_shape(
g
Adam/decay/readIdentity
Adam/decay*
T0*
_class
loc:@Adam/decay*
_output_shapes
: 
š
conv2d_10_targetPlaceholder*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*?
shape6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0
s
conv2d_10_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙

loss/conv2d_10_loss/subSubconv2d_10/Sigmoidconv2d_10_target*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
T0
s
loss/conv2d_10_loss/AbsAbsloss/conv2d_10_loss/sub*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
u
*loss/conv2d_10_loss/Mean/reduction_indicesConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ş
loss/conv2d_10_loss/MeanMeanloss/conv2d_10_loss/Abs*loss/conv2d_10_loss/Mean/reduction_indices*
	keep_dims( *

Tidx0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
}
,loss/conv2d_10_loss/Mean_1/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
ľ
loss/conv2d_10_loss/Mean_1Meanloss/conv2d_10_loss/Mean,loss/conv2d_10_loss/Mean_1/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *

Tidx0*
T0

loss/conv2d_10_loss/mulMulloss/conv2d_10_loss/Mean_1conv2d_10_sample_weights*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
loss/conv2d_10_loss/NotEqual/yConst*
_output_shapes
: *
valueB
 *    *
dtype0

loss/conv2d_10_loss/NotEqualNotEqualconv2d_10_sample_weightsloss/conv2d_10_loss/NotEqual/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

loss/conv2d_10_loss/CastCastloss/conv2d_10_loss/NotEqual*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0*

SrcT0
*
Truncate( 
c
loss/conv2d_10_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/conv2d_10_loss/Mean_2Meanloss/conv2d_10_loss/Castloss/conv2d_10_loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

loss/conv2d_10_loss/truedivRealDivloss/conv2d_10_loss/mulloss/conv2d_10_loss/Mean_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
loss/conv2d_10_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/conv2d_10_loss/Mean_3Meanloss/conv2d_10_loss/truedivloss/conv2d_10_loss/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
O

loss/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
X
loss/mulMul
loss/mul/xloss/conv2d_10_loss/Mean_3*
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
!training/Adam/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?*
_class
loc:@loss/mul
ś
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*

index_type0*
_class
loc:@loss/mul*
_output_shapes
: 
¨
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/conv2d_10_loss/Mean_3*
T0*
_class
loc:@loss/mul*
_output_shapes
: 

+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
ž
Etraining/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Reshape/shapeConst*
valueB:*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
dtype0*
_output_shapes
:
 
?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Etraining/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3
Ç
=training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/ShapeShapeloss/conv2d_10_loss/truediv*
T0*
out_type0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
_output_shapes
:
ł
<training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/TileTile?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Reshape=training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Shape*

Tmultiples0*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Shape_1Shapeloss/conv2d_10_loss/truediv*
T0*
out_type0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
_output_shapes
:
ą
?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Shape_2Const*
_output_shapes
: *
valueB *-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
dtype0
ś
=training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/ConstConst*
valueB: *-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
dtype0*
_output_shapes
:
ą
<training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/ProdProd?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Shape_1=training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Const*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
_output_shapes
: *

Tidx0*
	keep_dims( 
¸
?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Const_1Const*
valueB: *-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
dtype0*
_output_shapes
:
ľ
>training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Prod_1Prod?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Shape_2?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Const_1*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
˛
Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Maximum/yConst*
value	B :*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
dtype0*
_output_shapes
: 

?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/MaximumMaximum>training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Prod_1Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Maximum/y*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
_output_shapes
: 

@training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/floordivFloorDiv<training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Prod?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Maximum*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
_output_shapes
: *
T0
ő
<training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/CastCast@training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3
Ł
?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/truedivRealDiv<training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Tile<training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Cast*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
>training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/ShapeShapeloss/conv2d_10_loss/mul*
_output_shapes
:*
T0*
out_type0*.
_class$
" loc:@loss/conv2d_10_loss/truediv
ł
@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Shape_1Const*
valueB *.
_class$
" loc:@loss/conv2d_10_loss/truediv*
dtype0*
_output_shapes
: 
Ö
Ntraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs>training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Shape@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Shape_1*.
_class$
" loc:@loss/conv2d_10_loss/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/RealDivRealDiv?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/truedivloss/conv2d_10_loss/Mean_2*.
_class$
" loc:@loss/conv2d_10_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ĺ
<training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/SumSum@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/RealDivNtraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/BroadcastGradientArgs*
T0*.
_class$
" loc:@loss/conv2d_10_loss/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
ľ
@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/ReshapeReshape<training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Sum>training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Shape*
T0*
Tshape0*.
_class$
" loc:@loss/conv2d_10_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
<training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/NegNegloss/conv2d_10_loss/mul*
T0*.
_class$
" loc:@loss/conv2d_10_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Btraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/RealDiv_1RealDiv<training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Negloss/conv2d_10_loss/Mean_2*
T0*.
_class$
" loc:@loss/conv2d_10_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Btraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/RealDiv_2RealDivBtraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/RealDiv_1loss/conv2d_10_loss/Mean_2*
T0*.
_class$
" loc:@loss/conv2d_10_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
<training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/mulMul?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/truedivBtraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/RealDiv_2*
T0*.
_class$
" loc:@loss/conv2d_10_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
>training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Sum_1Sum<training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/mulPtraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/BroadcastGradientArgs:1*
T0*.
_class$
" loc:@loss/conv2d_10_loss/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ž
Btraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Reshape_1Reshape>training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Sum_1@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Shape_1*
T0*
Tshape0*.
_class$
" loc:@loss/conv2d_10_loss/truediv*
_output_shapes
: 
Ŕ
:training/Adam/gradients/loss/conv2d_10_loss/mul_grad/ShapeShapeloss/conv2d_10_loss/Mean_1*
T0*
out_type0**
_class 
loc:@loss/conv2d_10_loss/mul*
_output_shapes
:
Ŕ
<training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Shape_1Shapeconv2d_10_sample_weights*
_output_shapes
:*
T0*
out_type0**
_class 
loc:@loss/conv2d_10_loss/mul
Ć
Jtraining/Adam/gradients/loss/conv2d_10_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Shape<training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Shape_1*
T0**
_class 
loc:@loss/conv2d_10_loss/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ő
8training/Adam/gradients/loss/conv2d_10_loss/mul_grad/MulMul@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Reshapeconv2d_10_sample_weights*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0**
_class 
loc:@loss/conv2d_10_loss/mul
ą
8training/Adam/gradients/loss/conv2d_10_loss/mul_grad/SumSum8training/Adam/gradients/loss/conv2d_10_loss/mul_grad/MulJtraining/Adam/gradients/loss/conv2d_10_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/conv2d_10_loss/mul
Ľ
<training/Adam/gradients/loss/conv2d_10_loss/mul_grad/ReshapeReshape8training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Sum:training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0**
_class 
loc:@loss/conv2d_10_loss/mul
ů
:training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Mul_1Mulloss/conv2d_10_loss/Mean_1@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Reshape*
T0**
_class 
loc:@loss/conv2d_10_loss/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
:training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Sum_1Sum:training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Mul_1Ltraining/Adam/gradients/loss/conv2d_10_loss/mul_grad/BroadcastGradientArgs:1*
T0**
_class 
loc:@loss/conv2d_10_loss/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ť
>training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Reshape_1Reshape:training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Sum_1<training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Shape_1*
Tshape0**
_class 
loc:@loss/conv2d_10_loss/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ä
=training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/ShapeShapeloss/conv2d_10_loss/Mean*
_output_shapes
:*
T0*
out_type0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1
­
<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/SizeConst*
value	B :*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
dtype0*
_output_shapes
: 

;training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/addAdd,loss/conv2d_10_loss/Mean_1/reduction_indices<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Size*
_output_shapes
:*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1

;training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/modFloorMod;training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/add<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Size*
_output_shapes
:*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1
¸
?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape_1Const*
valueB:*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
dtype0*
_output_shapes
:
´
Ctraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/range/startConst*
value	B : *-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
dtype0*
_output_shapes
: 
´
Ctraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/range/deltaConst*
value	B :*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
dtype0*
_output_shapes
: 
ĺ
=training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/rangeRangeCtraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/range/start<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/SizeCtraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/range/delta*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
_output_shapes
:*

Tidx0
ł
Btraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1
Ż
<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/FillFill?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape_1Btraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Fill/value*
T0*

index_type0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
_output_shapes
:
Ź
Etraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/DynamicStitchDynamicStitch=training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/range;training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/mod=training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Fill*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
N*
_output_shapes
:
˛
Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Maximum/yConst*
value	B :*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
dtype0*
_output_shapes
: 
¨
?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/MaximumMaximumEtraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/DynamicStitchAtraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Maximum/y*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
_output_shapes
:
 
@training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/floordivFloorDiv=training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Maximum*
_output_shapes
:*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1
Ô
?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/ReshapeReshape<training/Adam/gradients/loss/conv2d_10_loss/mul_grad/ReshapeEtraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/DynamicStitch*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*
Tshape0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1
Đ
<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/TileTile?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Reshape@training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/floordiv*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*

Tmultiples0
Ć
?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape_2Shapeloss/conv2d_10_loss/Mean*
T0*
out_type0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
_output_shapes
:
Č
?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape_3Shapeloss/conv2d_10_loss/Mean_1*
T0*
out_type0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
_output_shapes
:
ś
=training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/ConstConst*
_output_shapes
:*
valueB: *-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
dtype0
ą
<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/ProdProd?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape_2=training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1
¸
?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *-
_class#
!loc:@loss/conv2d_10_loss/Mean_1
ľ
>training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Prod_1Prod?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape_3?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1
´
Ctraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Maximum_1/yConst*
value	B :*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
dtype0*
_output_shapes
: 
Ą
Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Maximum_1Maximum>training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Prod_1Ctraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Maximum_1/y*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
_output_shapes
: 

Btraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/floordiv_1FloorDiv<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/ProdAtraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Maximum_1*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
_output_shapes
: 
÷
<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/CastCastBtraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/floordiv_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1
­
?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/truedivRealDiv<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Tile<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Cast*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*-
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
ż
;training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/ShapeShapeloss/conv2d_10_loss/Abs*
T0*
out_type0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
:
Š
:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/SizeConst*
value	B :*+
_class!
loc:@loss/conv2d_10_loss/Mean*
dtype0*
_output_shapes
: 
ö
9training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/addAdd*loss/conv2d_10_loss/Mean/reduction_indices:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Size*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
: 

9training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/modFloorMod9training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/add:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Size*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
: 
­
=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape_1Const*
valueB *+
_class!
loc:@loss/conv2d_10_loss/Mean*
dtype0*
_output_shapes
: 
°
Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/range/startConst*
value	B : *+
_class!
loc:@loss/conv2d_10_loss/Mean*
dtype0*
_output_shapes
: 
°
Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/range/deltaConst*
value	B :*+
_class!
loc:@loss/conv2d_10_loss/Mean*
dtype0*
_output_shapes
: 
Ű
;training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/rangeRangeAtraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/range/start:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/SizeAtraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/range/delta*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
:*

Tidx0
Ż
@training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*+
_class!
loc:@loss/conv2d_10_loss/Mean
Ł
:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/FillFill=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape_1@training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Fill/value*
T0*

index_type0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
: 
 
Ctraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/DynamicStitchDynamicStitch;training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/range9training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/mod;training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Fill*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
N*
_output_shapes
:
Ž
?training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Maximum/yConst*
value	B :*+
_class!
loc:@loss/conv2d_10_loss/Mean*
dtype0*
_output_shapes
: 
 
=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/MaximumMaximumCtraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/DynamicStitch?training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Maximum/y*
_output_shapes
:*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean

>training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/floordivFloorDiv;training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Maximum*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
:
Ţ
=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/ReshapeReshape?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/truedivCtraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*+
_class!
loc:@loss/conv2d_10_loss/Mean*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ő
:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/TileTile=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Reshape>training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/floordiv*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*

Tmultiples0
Á
=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape_2Shapeloss/conv2d_10_loss/Abs*
T0*
out_type0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
:
Â
=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape_3Shapeloss/conv2d_10_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
:
˛
;training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/ConstConst*
valueB: *+
_class!
loc:@loss/conv2d_10_loss/Mean*
dtype0*
_output_shapes
:
Š
:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/ProdProd=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape_2;training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean
´
=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Const_1Const*
valueB: *+
_class!
loc:@loss/conv2d_10_loss/Mean*
dtype0*
_output_shapes
:
­
<training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Prod_1Prod=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape_3=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Const_1*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
°
Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Maximum_1/yConst*
value	B :*+
_class!
loc:@loss/conv2d_10_loss/Mean*
dtype0*
_output_shapes
: 

?training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Maximum_1Maximum<training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Prod_1Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Maximum_1/y*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
: 

@training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/floordiv_1FloorDiv:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Prod?training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Maximum_1*
_output_shapes
: *
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean
ń
:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/CastCast@training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
Truncate( 
Š
=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/truedivRealDiv:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Tile:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Cast*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
Â
9training/Adam/gradients/loss/conv2d_10_loss/Abs_grad/SignSignloss/conv2d_10_loss/sub*
T0**
_class 
loc:@loss/conv2d_10_loss/Abs*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
Ą
8training/Adam/gradients/loss/conv2d_10_loss/Abs_grad/mulMul=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/truediv9training/Adam/gradients/loss/conv2d_10_loss/Abs_grad/Sign*
T0**
_class 
loc:@loss/conv2d_10_loss/Abs*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
ˇ
:training/Adam/gradients/loss/conv2d_10_loss/sub_grad/ShapeShapeconv2d_10/Sigmoid*
_output_shapes
:*
T0*
out_type0**
_class 
loc:@loss/conv2d_10_loss/sub
¸
<training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Shape_1Shapeconv2d_10_target*
T0*
out_type0**
_class 
loc:@loss/conv2d_10_loss/sub*
_output_shapes
:
Ć
Jtraining/Adam/gradients/loss/conv2d_10_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs:training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Shape<training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Shape_1*
T0**
_class 
loc:@loss/conv2d_10_loss/sub*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ą
8training/Adam/gradients/loss/conv2d_10_loss/sub_grad/SumSum8training/Adam/gradients/loss/conv2d_10_loss/Abs_grad/mulJtraining/Adam/gradients/loss/conv2d_10_loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/conv2d_10_loss/sub
ł
<training/Adam/gradients/loss/conv2d_10_loss/sub_grad/ReshapeReshape8training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Sum:training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Shape*
T0*
Tshape0**
_class 
loc:@loss/conv2d_10_loss/sub*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
ľ
:training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Sum_1Sum8training/Adam/gradients/loss/conv2d_10_loss/Abs_grad/mulLtraining/Adam/gradients/loss/conv2d_10_loss/sub_grad/BroadcastGradientArgs:1*
T0**
_class 
loc:@loss/conv2d_10_loss/sub*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ę
8training/Adam/gradients/loss/conv2d_10_loss/sub_grad/NegNeg:training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Sum_1*
_output_shapes
:*
T0**
_class 
loc:@loss/conv2d_10_loss/sub
Đ
>training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Reshape_1Reshape8training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Neg<training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Shape_1*
T0*
Tshape0**
_class 
loc:@loss/conv2d_10_loss/sub*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ü
:training/Adam/gradients/conv2d_10/Sigmoid_grad/SigmoidGradSigmoidGradconv2d_10/Sigmoid<training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Reshape*
T0*$
_class
loc:@conv2d_10/Sigmoid*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
ç
:training/Adam/gradients/conv2d_10/BiasAdd_grad/BiasAddGradBiasAddGrad:training/Adam/gradients/conv2d_10/Sigmoid_grad/SigmoidGrad*
T0*$
_class
loc:@conv2d_10/BiasAdd*
data_formatNHWC*
_output_shapes
:
×
9training/Adam/gradients/conv2d_10/convolution_grad/ShapeNShapeNconv2d_9/Reluconv2d_10/kernel/read*
T0*
out_type0*(
_class
loc:@conv2d_10/convolution*
N* 
_output_shapes
::
Ŕ
Ftraining/Adam/gradients/conv2d_10/convolution_grad/Conv2DBackpropInputConv2DBackpropInput9training/Adam/gradients/conv2d_10/convolution_grad/ShapeNconv2d_10/kernel/read:training/Adam/gradients/conv2d_10/Sigmoid_grad/SigmoidGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
	dilations
*
T0*(
_class
loc:@conv2d_10/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
ą
Gtraining/Adam/gradients/conv2d_10/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_9/Relu;training/Adam/gradients/conv2d_10/convolution_grad/ShapeN:1:training/Adam/gradients/conv2d_10/Sigmoid_grad/SigmoidGrad*&
_output_shapes
:@*
	dilations
*
T0*(
_class
loc:@conv2d_10/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
ô
3training/Adam/gradients/conv2d_9/Relu_grad/ReluGradReluGradFtraining/Adam/gradients/conv2d_10/convolution_grad/Conv2DBackpropInputconv2d_9/Relu* 
_class
loc:@conv2d_9/Relu*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0
Ţ
9training/Adam/gradients/conv2d_9/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_9/Relu_grad/ReluGrad*#
_class
loc:@conv2d_9/BiasAdd*
data_formatNHWC*
_output_shapes
:@*
T0
Ô
8training/Adam/gradients/conv2d_9/convolution_grad/ShapeNShapeNconv2d_8/Reluconv2d_9/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0*'
_class
loc:@conv2d_9/convolution
´
Etraining/Adam/gradients/conv2d_9/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_9/convolution_grad/ShapeNconv2d_9/kernel/read3training/Adam/gradients/conv2d_9/Relu_grad/ReluGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
	dilations
*
T0*'
_class
loc:@conv2d_9/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Ś
Ftraining/Adam/gradients/conv2d_9/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_8/Relu:training/Adam/gradients/conv2d_9/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_9/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_9/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@@
ó
3training/Adam/gradients/conv2d_8/Relu_grad/ReluGradReluGradEtraining/Adam/gradients/conv2d_9/convolution_grad/Conv2DBackpropInputconv2d_8/Relu*
T0* 
_class
loc:@conv2d_8/Relu*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
Ţ
9training/Adam/gradients/conv2d_8/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_8/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_8/BiasAdd*
data_formatNHWC*
_output_shapes
:@
Ű
8training/Adam/gradients/conv2d_8/convolution_grad/ShapeNShapeNconcatenate_4/concatconv2d_8/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_8/convolution*
N* 
_output_shapes
::
ľ
Etraining/Adam/gradients/conv2d_8/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_8/convolution_grad/ShapeNconv2d_8/kernel/read3training/Adam/gradients/conv2d_8/Relu_grad/ReluGrad*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¸Đ*
	dilations
*
T0*'
_class
loc:@conv2d_8/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ž
Ftraining/Adam/gradients/conv2d_8/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconcatenate_4/concat:training/Adam/gradients/conv2d_8/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_8/Relu_grad/ReluGrad*
T0*'
_class
loc:@conv2d_8/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations

Ą
6training/Adam/gradients/concatenate_4/concat_grad/RankConst*
value	B :*'
_class
loc:@concatenate_4/concat*
dtype0*
_output_shapes
: 
Ţ
5training/Adam/gradients/concatenate_4/concat_grad/modFloorModconcatenate_4/concat/axis6training/Adam/gradients/concatenate_4/concat_grad/Rank*
_output_shapes
: *
T0*'
_class
loc:@concatenate_4/concat
Ť
7training/Adam/gradients/concatenate_4/concat_grad/ShapeShapeconv1b/Relu*
out_type0*'
_class
loc:@concatenate_4/concat*
_output_shapes
:*
T0
Ë
8training/Adam/gradients/concatenate_4/concat_grad/ShapeNShapeNconv1b/Reluconv2d_7/Relu*
T0*
out_type0*'
_class
loc:@concatenate_4/concat*
N* 
_output_shapes
::
Ď
>training/Adam/gradients/concatenate_4/concat_grad/ConcatOffsetConcatOffset5training/Adam/gradients/concatenate_4/concat_grad/mod8training/Adam/gradients/concatenate_4/concat_grad/ShapeN:training/Adam/gradients/concatenate_4/concat_grad/ShapeN:1*'
_class
loc:@concatenate_4/concat*
N* 
_output_shapes
::
ó
7training/Adam/gradients/concatenate_4/concat_grad/SliceSliceEtraining/Adam/gradients/conv2d_8/convolution_grad/Conv2DBackpropInput>training/Adam/gradients/concatenate_4/concat_grad/ConcatOffset8training/Adam/gradients/concatenate_4/concat_grad/ShapeN*'
_class
loc:@concatenate_4/concat*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0*
Index0
ů
9training/Adam/gradients/concatenate_4/concat_grad/Slice_1SliceEtraining/Adam/gradients/conv2d_8/convolution_grad/Conv2DBackpropInput@training/Adam/gradients/concatenate_4/concat_grad/ConcatOffset:1:training/Adam/gradients/concatenate_4/concat_grad/ShapeN:1*
T0*
Index0*'
_class
loc:@concatenate_4/concat*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
ç
3training/Adam/gradients/conv2d_7/Relu_grad/ReluGradReluGrad9training/Adam/gradients/concatenate_4/concat_grad/Slice_1conv2d_7/Relu*
T0* 
_class
loc:@conv2d_7/Relu*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
Ţ
9training/Adam/gradients/conv2d_7/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_7/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0*#
_class
loc:@conv2d_7/BiasAdd
ě
8training/Adam/gradients/conv2d_7/convolution_grad/ShapeNShapeN%up_sampling2d_3/ResizeNearestNeighborconv2d_7/kernel/read*
out_type0*'
_class
loc:@conv2d_7/convolution*
N* 
_output_shapes
::*
T0
ľ
Etraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_7/convolution_grad/ShapeNconv2d_7/kernel/read3training/Adam/gradients/conv2d_7/Relu_grad/ReluGrad*'
_class
loc:@conv2d_7/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¸Đ*
	dilations
*
T0
ż
Ftraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_3/ResizeNearestNeighbor:training/Adam/gradients/conv2d_7/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_7/Relu_grad/ReluGrad*'
_class
loc:@conv2d_7/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0
ě
atraining/Adam/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"  (  *8
_class.
,*loc:@up_sampling2d_3/ResizeNearestNeighbor*
dtype0*
_output_shapes
:
Ż
\training/Adam/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradEtraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropInputatraining/Adam/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
align_corners( *
T0*8
_class.
,*loc:@up_sampling2d_3/ResizeNearestNeighbor

3training/Adam/gradients/conv2d_6/Relu_grad/ReluGradReluGrad\training/Adam/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradconv2d_6/Relu*
T0* 
_class
loc:@conv2d_6/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
ß
9training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_6/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_6/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ô
8training/Adam/gradients/conv2d_6/convolution_grad/ShapeNShapeNconv2d_5/Reluconv2d_6/kernel/read* 
_output_shapes
::*
T0*
out_type0*'
_class
loc:@conv2d_6/convolution*
N
ľ
Etraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_6/convolution_grad/ShapeNconv2d_6/kernel/read3training/Adam/gradients/conv2d_6/Relu_grad/ReluGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
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
¨
Ftraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_5/Relu:training/Adam/gradients/conv2d_6/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_6/Relu_grad/ReluGrad*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0*'
_class
loc:@conv2d_6/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
ô
3training/Adam/gradients/conv2d_5/Relu_grad/ReluGradReluGradEtraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputconv2d_5/Relu*
T0* 
_class
loc:@conv2d_5/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
ß
9training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_5/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_5/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ű
8training/Adam/gradients/conv2d_5/convolution_grad/ShapeNShapeNconcatenate_3/concatconv2d_5/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_5/convolution*
N* 
_output_shapes
::
ľ
Etraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_5/convolution_grad/ShapeNconv2d_5/kernel/read3training/Adam/gradients/conv2d_5/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_5/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
Ż
Ftraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconcatenate_3/concat:training/Adam/gradients/conv2d_5/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_5/Relu_grad/ReluGrad*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0*'
_class
loc:@conv2d_5/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
Ą
6training/Adam/gradients/concatenate_3/concat_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :*'
_class
loc:@concatenate_3/concat
Ţ
5training/Adam/gradients/concatenate_3/concat_grad/modFloorModconcatenate_3/concat/axis6training/Adam/gradients/concatenate_3/concat_grad/Rank*
T0*'
_class
loc:@concatenate_3/concat*
_output_shapes
: 
Ť
7training/Adam/gradients/concatenate_3/concat_grad/ShapeShapeconv2b/Relu*
T0*
out_type0*'
_class
loc:@concatenate_3/concat*
_output_shapes
:
Ë
8training/Adam/gradients/concatenate_3/concat_grad/ShapeNShapeNconv2b/Reluconv2d_4/Relu*
T0*
out_type0*'
_class
loc:@concatenate_3/concat*
N* 
_output_shapes
::
Ď
>training/Adam/gradients/concatenate_3/concat_grad/ConcatOffsetConcatOffset5training/Adam/gradients/concatenate_3/concat_grad/mod8training/Adam/gradients/concatenate_3/concat_grad/ShapeN:training/Adam/gradients/concatenate_3/concat_grad/ShapeN:1*
N* 
_output_shapes
::*'
_class
loc:@concatenate_3/concat
ô
7training/Adam/gradients/concatenate_3/concat_grad/SliceSliceEtraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropInput>training/Adam/gradients/concatenate_3/concat_grad/ConcatOffset8training/Adam/gradients/concatenate_3/concat_grad/ShapeN*
T0*
Index0*'
_class
loc:@concatenate_3/concat*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
ú
9training/Adam/gradients/concatenate_3/concat_grad/Slice_1SliceEtraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropInput@training/Adam/gradients/concatenate_3/concat_grad/ConcatOffset:1:training/Adam/gradients/concatenate_3/concat_grad/ShapeN:1*
T0*
Index0*'
_class
loc:@concatenate_3/concat*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
č
3training/Adam/gradients/conv2d_4/Relu_grad/ReluGradReluGrad9training/Adam/gradients/concatenate_3/concat_grad/Slice_1conv2d_4/Relu*
T0* 
_class
loc:@conv2d_4/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
ß
9training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_4/Relu_grad/ReluGrad*#
_class
loc:@conv2d_4/BiasAdd*
data_formatNHWC*
_output_shapes	
:*
T0
ě
8training/Adam/gradients/conv2d_4/convolution_grad/ShapeNShapeN%up_sampling2d_2/ResizeNearestNeighborconv2d_4/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_4/convolution*
N* 
_output_shapes
::
ľ
Etraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_4/convolution_grad/ShapeNconv2d_4/kernel/read3training/Adam/gradients/conv2d_4/Relu_grad/ReluGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
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
Ŕ
Ftraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_2/ResizeNearestNeighbor:training/Adam/gradients/conv2d_4/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_4/Relu_grad/ReluGrad*(
_output_shapes
:*
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
ě
atraining/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"    *8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor*
dtype0*
_output_shapes
:
Ż
\training/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradEtraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputatraining/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
align_corners( *
T0*8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor

3training/Adam/gradients/conv2d_3/Relu_grad/ReluGradReluGrad\training/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradconv2d_3/Relu*
T0* 
_class
loc:@conv2d_3/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
ß
9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_3/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_3/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ô
8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNShapeNconv2d_2/Reluconv2d_3/kernel/read* 
_output_shapes
::*
T0*
out_type0*'
_class
loc:@conv2d_3/convolution*
N
ľ
Etraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNconv2d_3/kernel/read3training/Adam/gradients/conv2d_3/Relu_grad/ReluGrad*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*'
_class
loc:@conv2d_3/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
¨
Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_2/Relu:training/Adam/gradients/conv2d_3/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_3/Relu_grad/ReluGrad*'
_class
loc:@conv2d_3/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0
ô
3training/Adam/gradients/conv2d_2/Relu_grad/ReluGradReluGradEtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputconv2d_2/Relu*
T0* 
_class
loc:@conv2d_2/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
ß
9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_2/Relu_grad/ReluGrad*#
_class
loc:@conv2d_2/BiasAdd*
data_formatNHWC*
_output_shapes	
:*
T0
Ű
8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNShapeNconcatenate_2/concatconv2d_2/kernel/read* 
_output_shapes
::*
T0*
out_type0*'
_class
loc:@conv2d_2/convolution*
N
ľ
Etraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNconv2d_2/kernel/read3training/Adam/gradients/conv2d_2/Relu_grad/ReluGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
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
Ż
Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconcatenate_2/concat:training/Adam/gradients/conv2d_2/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_2/Relu_grad/ReluGrad*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
Ą
6training/Adam/gradients/concatenate_2/concat_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :*'
_class
loc:@concatenate_2/concat
Ţ
5training/Adam/gradients/concatenate_2/concat_grad/modFloorModconcatenate_2/concat/axis6training/Adam/gradients/concatenate_2/concat_grad/Rank*
_output_shapes
: *
T0*'
_class
loc:@concatenate_2/concat
Ť
7training/Adam/gradients/concatenate_2/concat_grad/ShapeShapeconv3b/Relu*
_output_shapes
:*
T0*
out_type0*'
_class
loc:@concatenate_2/concat
Ë
8training/Adam/gradients/concatenate_2/concat_grad/ShapeNShapeNconv3b/Reluconv2d_1/Relu*
T0*
out_type0*'
_class
loc:@concatenate_2/concat*
N* 
_output_shapes
::
Ď
>training/Adam/gradients/concatenate_2/concat_grad/ConcatOffsetConcatOffset5training/Adam/gradients/concatenate_2/concat_grad/mod8training/Adam/gradients/concatenate_2/concat_grad/ShapeN:training/Adam/gradients/concatenate_2/concat_grad/ShapeN:1*'
_class
loc:@concatenate_2/concat*
N* 
_output_shapes
::
ô
7training/Adam/gradients/concatenate_2/concat_grad/SliceSliceEtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput>training/Adam/gradients/concatenate_2/concat_grad/ConcatOffset8training/Adam/gradients/concatenate_2/concat_grad/ShapeN*
T0*
Index0*'
_class
loc:@concatenate_2/concat*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
ú
9training/Adam/gradients/concatenate_2/concat_grad/Slice_1SliceEtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput@training/Adam/gradients/concatenate_2/concat_grad/ConcatOffset:1:training/Adam/gradients/concatenate_2/concat_grad/ShapeN:1*'
_class
loc:@concatenate_2/concat*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0*
Index0
č
3training/Adam/gradients/conv2d_1/Relu_grad/ReluGradReluGrad9training/Adam/gradients/concatenate_2/concat_grad/Slice_1conv2d_1/Relu*
T0* 
_class
loc:@conv2d_1/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
ß
9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_1/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_1/BiasAdd*
data_formatNHWC*
_output_shapes	
:
ě
8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNShapeN%up_sampling2d_1/ResizeNearestNeighborconv2d_1/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_1/convolution*
N* 
_output_shapes
::
ľ
Etraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNconv2d_1/kernel/read3training/Adam/gradients/conv2d_1/Relu_grad/ReluGrad*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ŕ
Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_1/ResizeNearestNeighbor:training/Adam/gradients/conv2d_1/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_1/Relu_grad/ReluGrad*
T0*'
_class
loc:@conv2d_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:*
	dilations

ě
atraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"   Ę   *8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*
dtype0*
_output_shapes
:
Ż
\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradEtraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputatraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
T0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
align_corners( 

1training/Adam/gradients/conv7b/Relu_grad/ReluGradReluGrad\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradconv7b/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0*
_class
loc:@conv7b/Relu
Ů
7training/Adam/gradients/conv7b/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv7b/Relu_grad/ReluGrad*
T0*!
_class
loc:@conv7b/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ě
6training/Adam/gradients/conv7b/convolution_grad/ShapeNShapeNconv7a/Reluconv7b/kernel/read*
T0*
out_type0*%
_class
loc:@conv7b/convolution*
N* 
_output_shapes
::
Ť
Ctraining/Adam/gradients/conv7b/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv7b/convolution_grad/ShapeNconv7b/kernel/read1training/Adam/gradients/conv7b/Relu_grad/ReluGrad*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations
*
T0*%
_class
loc:@conv7b/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

Dtraining/Adam/gradients/conv7b/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv7a/Relu8training/Adam/gradients/conv7b/convolution_grad/ShapeN:11training/Adam/gradients/conv7b/Relu_grad/ReluGrad*
	dilations
*
T0*%
_class
loc:@conv7b/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:
ě
1training/Adam/gradients/conv7a/Relu_grad/ReluGradReluGradCtraining/Adam/gradients/conv7b/convolution_grad/Conv2DBackpropInputconv7a/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0*
_class
loc:@conv7a/Relu
Ů
7training/Adam/gradients/conv7a/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv7a/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0*!
_class
loc:@conv7a/BiasAdd
Ő
6training/Adam/gradients/conv7a/convolution_grad/ShapeNShapeNconcatenate_1/concatconv7a/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0*%
_class
loc:@conv7a/convolution
Ť
Ctraining/Adam/gradients/conv7a/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv7a/convolution_grad/ShapeNconv7a/kernel/read1training/Adam/gradients/conv7a/Relu_grad/ReluGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations
*
T0*%
_class
loc:@conv7a/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
§
Dtraining/Adam/gradients/conv7a/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconcatenate_1/concat8training/Adam/gradients/conv7a/convolution_grad/ShapeN:11training/Adam/gradients/conv7a/Relu_grad/ReluGrad*
T0*%
_class
loc:@conv7a/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:*
	dilations

Ą
6training/Adam/gradients/concatenate_1/concat_grad/RankConst*
value	B :*'
_class
loc:@concatenate_1/concat*
dtype0*
_output_shapes
: 
Ţ
5training/Adam/gradients/concatenate_1/concat_grad/modFloorModconcatenate_1/concat/axis6training/Adam/gradients/concatenate_1/concat_grad/Rank*
T0*'
_class
loc:@concatenate_1/concat*
_output_shapes
: 
°
7training/Adam/gradients/concatenate_1/concat_grad/ShapeShapedrop4/cond/Merge*
T0*
out_type0*'
_class
loc:@concatenate_1/concat*
_output_shapes
:
×
8training/Adam/gradients/concatenate_1/concat_grad/ShapeNShapeNdrop4/cond/Mergezero_padding2d_1/Pad*
N* 
_output_shapes
::*
T0*
out_type0*'
_class
loc:@concatenate_1/concat
Ď
>training/Adam/gradients/concatenate_1/concat_grad/ConcatOffsetConcatOffset5training/Adam/gradients/concatenate_1/concat_grad/mod8training/Adam/gradients/concatenate_1/concat_grad/ShapeN:training/Adam/gradients/concatenate_1/concat_grad/ShapeN:1*'
_class
loc:@concatenate_1/concat*
N* 
_output_shapes
::
ň
7training/Adam/gradients/concatenate_1/concat_grad/SliceSliceCtraining/Adam/gradients/conv7a/convolution_grad/Conv2DBackpropInput>training/Adam/gradients/concatenate_1/concat_grad/ConcatOffset8training/Adam/gradients/concatenate_1/concat_grad/ShapeN*
T0*
Index0*'
_class
loc:@concatenate_1/concat*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
ř
9training/Adam/gradients/concatenate_1/concat_grad/Slice_1SliceCtraining/Adam/gradients/conv7a/convolution_grad/Conv2DBackpropInput@training/Adam/gradients/concatenate_1/concat_grad/ConcatOffset:1:training/Adam/gradients/concatenate_1/concat_grad/ShapeN:1*
T0*
Index0*'
_class
loc:@concatenate_1/concat*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
Ą
6training/Adam/gradients/zero_padding2d_1/Pad_grad/RankConst*
_output_shapes
: *
value	B :*'
_class
loc:@zero_padding2d_1/Pad*
dtype0
¤
9training/Adam/gradients/zero_padding2d_1/Pad_grad/stack/1Const*
dtype0*
_output_shapes
: *
value	B :*'
_class
loc:@zero_padding2d_1/Pad

7training/Adam/gradients/zero_padding2d_1/Pad_grad/stackPack6training/Adam/gradients/zero_padding2d_1/Pad_grad/Rank9training/Adam/gradients/zero_padding2d_1/Pad_grad/stack/1*
T0*

axis *'
_class
loc:@zero_padding2d_1/Pad*
N*
_output_shapes
:
ˇ
=training/Adam/gradients/zero_padding2d_1/Pad_grad/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB"        *'
_class
loc:@zero_padding2d_1/Pad
ś
7training/Adam/gradients/zero_padding2d_1/Pad_grad/SliceSlicezero_padding2d_1/Pad/paddings=training/Adam/gradients/zero_padding2d_1/Pad_grad/Slice/begin7training/Adam/gradients/zero_padding2d_1/Pad_grad/stack*
T0*
Index0*'
_class
loc:@zero_padding2d_1/Pad*
_output_shapes

:
ť
?training/Adam/gradients/zero_padding2d_1/Pad_grad/Reshape/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*'
_class
loc:@zero_padding2d_1/Pad*
dtype0*
_output_shapes
:

9training/Adam/gradients/zero_padding2d_1/Pad_grad/ReshapeReshape7training/Adam/gradients/zero_padding2d_1/Pad_grad/Slice?training/Adam/gradients/zero_padding2d_1/Pad_grad/Reshape/shape*
T0*
Tshape0*'
_class
loc:@zero_padding2d_1/Pad*
_output_shapes
:
Ş
7training/Adam/gradients/zero_padding2d_1/Pad_grad/ShapeShape
conv6/Relu*
out_type0*'
_class
loc:@zero_padding2d_1/Pad*
_output_shapes
:*
T0
ä
9training/Adam/gradients/zero_padding2d_1/Pad_grad/Slice_1Slice9training/Adam/gradients/concatenate_1/concat_grad/Slice_19training/Adam/gradients/zero_padding2d_1/Pad_grad/Reshape7training/Adam/gradients/zero_padding2d_1/Pad_grad/Shape*
T0*
Index0*'
_class
loc:@zero_padding2d_1/Pad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
ß
0training/Adam/gradients/conv6/Relu_grad/ReluGradReluGrad9training/Adam/gradients/zero_padding2d_1/Pad_grad/Slice_1
conv6/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0*
_class
loc:@conv6/Relu
Ö
6training/Adam/gradients/conv6/BiasAdd_grad/BiasAddGradBiasAddGrad0training/Adam/gradients/conv6/Relu_grad/ReluGrad*
T0* 
_class
loc:@conv6/BiasAdd*
data_formatNHWC*
_output_shapes	
:
×
5training/Adam/gradients/conv6/convolution_grad/ShapeNShapeNup1/ResizeNearestNeighborconv6/kernel/read*
T0*
out_type0*$
_class
loc:@conv6/convolution*
N* 
_output_shapes
::
Ś
Btraining/Adam/gradients/conv6/convolution_grad/Conv2DBackpropInputConv2DBackpropInput5training/Adam/gradients/conv6/convolution_grad/ShapeNconv6/kernel/read0training/Adam/gradients/conv6/Relu_grad/ReluGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations
*
T0*$
_class
loc:@conv6/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
¨
Ctraining/Adam/gradients/conv6/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterup1/ResizeNearestNeighbor7training/Adam/gradients/conv6/convolution_grad/ShapeN:10training/Adam/gradients/conv6/Relu_grad/ReluGrad*(
_output_shapes
:*
	dilations
*
T0*$
_class
loc:@conv6/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Ô
Utraining/Adam/gradients/up1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"C   e   *,
_class"
 loc:@up1/ResizeNearestNeighbor*
dtype0*
_output_shapes
:

Ptraining/Adam/gradients/up1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradBtraining/Adam/gradients/conv6/convolution_grad/Conv2DBackpropInputUtraining/Adam/gradients/up1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
align_corners( *
T0*,
_class"
 loc:@up1/ResizeNearestNeighbor*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
Ź
7training/Adam/gradients/drop5/cond/Merge_grad/cond_gradSwitchPtraining/Adam/gradients/up1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGraddrop5/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙Ce:˙˙˙˙˙˙˙˙˙Ce*
T0*,
_class"
 loc:@up1/ResizeNearestNeighbor
Ŕ
training/Adam/gradients/SwitchSwitchconv5b/Reludrop5/cond/pred_id*
_class
loc:@conv5b/Relu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙Ce:˙˙˙˙˙˙˙˙˙Ce*
T0
Š
 training/Adam/gradients/IdentityIdentity training/Adam/gradients/Switch:1*
T0*
_class
loc:@conv5b/Relu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

training/Adam/gradients/Shape_1Shape training/Adam/gradients/Switch:1*
out_type0*
_class
loc:@conv5b/Relu*
_output_shapes
:*
T0
Ť
#training/Adam/gradients/zeros/ConstConst!^training/Adam/gradients/Identity*
valueB
 *    *
_class
loc:@conv5b/Relu*
dtype0*
_output_shapes
: 
Ř
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*
T0*

index_type0*
_class
loc:@conv5b/Relu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

:training/Adam/gradients/drop5/cond/Switch_1_grad/cond_gradMerge7training/Adam/gradients/drop5/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros*
T0*
_class
loc:@conv5b/Relu*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ce: 
ž
9training/Adam/gradients/drop5/cond/dropout/mul_grad/ShapeShapedrop5/cond/dropout/truediv*
out_type0*)
_class
loc:@drop5/cond/dropout/mul*
_output_shapes
:*
T0
ž
;training/Adam/gradients/drop5/cond/dropout/mul_grad/Shape_1Shapedrop5/cond/dropout/Floor*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@drop5/cond/dropout/mul
Â
Itraining/Adam/gradients/drop5/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/drop5/cond/dropout/mul_grad/Shape;training/Adam/gradients/drop5/cond/dropout/mul_grad/Shape_1*
T0*)
_class
loc:@drop5/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ů
7training/Adam/gradients/drop5/cond/dropout/mul_grad/MulMul9training/Adam/gradients/drop5/cond/Merge_grad/cond_grad:1drop5/cond/dropout/Floor*
T0*)
_class
loc:@drop5/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
­
7training/Adam/gradients/drop5/cond/dropout/mul_grad/SumSum7training/Adam/gradients/drop5/cond/dropout/mul_grad/MulItraining/Adam/gradients/drop5/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*)
_class
loc:@drop5/cond/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ž
;training/Adam/gradients/drop5/cond/dropout/mul_grad/ReshapeReshape7training/Adam/gradients/drop5/cond/dropout/mul_grad/Sum9training/Adam/gradients/drop5/cond/dropout/mul_grad/Shape*
T0*
Tshape0*)
_class
loc:@drop5/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
ý
9training/Adam/gradients/drop5/cond/dropout/mul_grad/Mul_1Muldrop5/cond/dropout/truediv9training/Adam/gradients/drop5/cond/Merge_grad/cond_grad:1*)
_class
loc:@drop5/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
T0
ł
9training/Adam/gradients/drop5/cond/dropout/mul_grad/Sum_1Sum9training/Adam/gradients/drop5/cond/dropout/mul_grad/Mul_1Ktraining/Adam/gradients/drop5/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*)
_class
loc:@drop5/cond/dropout/mul
´
=training/Adam/gradients/drop5/cond/dropout/mul_grad/Reshape_1Reshape9training/Adam/gradients/drop5/cond/dropout/mul_grad/Sum_1;training/Adam/gradients/drop5/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*)
_class
loc:@drop5/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
ş
=training/Adam/gradients/drop5/cond/dropout/truediv_grad/ShapeShapedrop5/cond/mul*
T0*
out_type0*-
_class#
!loc:@drop5/cond/dropout/truediv*
_output_shapes
:
ą
?training/Adam/gradients/drop5/cond/dropout/truediv_grad/Shape_1Const*
valueB *-
_class#
!loc:@drop5/cond/dropout/truediv*
dtype0*
_output_shapes
: 
Ň
Mtraining/Adam/gradients/drop5/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/drop5/cond/dropout/truediv_grad/Shape?training/Adam/gradients/drop5/cond/dropout/truediv_grad/Shape_1*-
_class#
!loc:@drop5/cond/dropout/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

?training/Adam/gradients/drop5/cond/dropout/truediv_grad/RealDivRealDiv;training/Adam/gradients/drop5/cond/dropout/mul_grad/Reshapedrop5/cond/dropout/sub*-
_class#
!loc:@drop5/cond/dropout/truediv*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
T0
Á
;training/Adam/gradients/drop5/cond/dropout/truediv_grad/SumSum?training/Adam/gradients/drop5/cond/dropout/truediv_grad/RealDivMtraining/Adam/gradients/drop5/cond/dropout/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@drop5/cond/dropout/truediv
ž
?training/Adam/gradients/drop5/cond/dropout/truediv_grad/ReshapeReshape;training/Adam/gradients/drop5/cond/dropout/truediv_grad/Sum=training/Adam/gradients/drop5/cond/dropout/truediv_grad/Shape*
Tshape0*-
_class#
!loc:@drop5/cond/dropout/truediv*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
T0
ź
;training/Adam/gradients/drop5/cond/dropout/truediv_grad/NegNegdrop5/cond/mul*
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

Atraining/Adam/gradients/drop5/cond/dropout/truediv_grad/RealDiv_1RealDiv;training/Adam/gradients/drop5/cond/dropout/truediv_grad/Negdrop5/cond/dropout/sub*
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

Atraining/Adam/gradients/drop5/cond/dropout/truediv_grad/RealDiv_2RealDivAtraining/Adam/gradients/drop5/cond/dropout/truediv_grad/RealDiv_1drop5/cond/dropout/sub*
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
Ź
;training/Adam/gradients/drop5/cond/dropout/truediv_grad/mulMul;training/Adam/gradients/drop5/cond/dropout/mul_grad/ReshapeAtraining/Adam/gradients/drop5/cond/dropout/truediv_grad/RealDiv_2*
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
Á
=training/Adam/gradients/drop5/cond/dropout/truediv_grad/Sum_1Sum;training/Adam/gradients/drop5/cond/dropout/truediv_grad/mulOtraining/Adam/gradients/drop5/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ş
Atraining/Adam/gradients/drop5/cond/dropout/truediv_grad/Reshape_1Reshape=training/Adam/gradients/drop5/cond/dropout/truediv_grad/Sum_1?training/Adam/gradients/drop5/cond/dropout/truediv_grad/Shape_1*
Tshape0*-
_class#
!loc:@drop5/cond/dropout/truediv*
_output_shapes
: *
T0
Ť
1training/Adam/gradients/drop5/cond/mul_grad/ShapeShapedrop5/cond/mul/Switch:1*
T0*
out_type0*!
_class
loc:@drop5/cond/mul*
_output_shapes
:

3training/Adam/gradients/drop5/cond/mul_grad/Shape_1Const*
valueB *!
_class
loc:@drop5/cond/mul*
dtype0*
_output_shapes
: 
˘
Atraining/Adam/gradients/drop5/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1training/Adam/gradients/drop5/cond/mul_grad/Shape3training/Adam/gradients/drop5/cond/mul_grad/Shape_1*!
_class
loc:@drop5/cond/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ç
/training/Adam/gradients/drop5/cond/mul_grad/MulMul?training/Adam/gradients/drop5/cond/dropout/truediv_grad/Reshapedrop5/cond/mul/y*
T0*!
_class
loc:@drop5/cond/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

/training/Adam/gradients/drop5/cond/mul_grad/SumSum/training/Adam/gradients/drop5/cond/mul_grad/MulAtraining/Adam/gradients/drop5/cond/mul_grad/BroadcastGradientArgs*!
_class
loc:@drop5/cond/mul*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

3training/Adam/gradients/drop5/cond/mul_grad/ReshapeReshape/training/Adam/gradients/drop5/cond/mul_grad/Sum1training/Adam/gradients/drop5/cond/mul_grad/Shape*
T0*
Tshape0*!
_class
loc:@drop5/cond/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
đ
1training/Adam/gradients/drop5/cond/mul_grad/Mul_1Muldrop5/cond/mul/Switch:1?training/Adam/gradients/drop5/cond/dropout/truediv_grad/Reshape*
T0*!
_class
loc:@drop5/cond/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

1training/Adam/gradients/drop5/cond/mul_grad/Sum_1Sum1training/Adam/gradients/drop5/cond/mul_grad/Mul_1Ctraining/Adam/gradients/drop5/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*!
_class
loc:@drop5/cond/mul
ú
5training/Adam/gradients/drop5/cond/mul_grad/Reshape_1Reshape1training/Adam/gradients/drop5/cond/mul_grad/Sum_13training/Adam/gradients/drop5/cond/mul_grad/Shape_1*
T0*
Tshape0*!
_class
loc:@drop5/cond/mul*
_output_shapes
: 
Â
 training/Adam/gradients/Switch_1Switchconv5b/Reludrop5/cond/pred_id*
T0*
_class
loc:@conv5b/Relu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙Ce:˙˙˙˙˙˙˙˙˙Ce
Ť
"training/Adam/gradients/Identity_1Identity training/Adam/gradients/Switch_1*
T0*
_class
loc:@conv5b/Relu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

training/Adam/gradients/Shape_2Shape training/Adam/gradients/Switch_1*
T0*
out_type0*
_class
loc:@conv5b/Relu*
_output_shapes
:
Ż
%training/Adam/gradients/zeros_1/ConstConst#^training/Adam/gradients/Identity_1*
valueB
 *    *
_class
loc:@conv5b/Relu*
dtype0*
_output_shapes
: 
Ü
training/Adam/gradients/zeros_1Filltraining/Adam/gradients/Shape_2%training/Adam/gradients/zeros_1/Const*
T0*

index_type0*
_class
loc:@conv5b/Relu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

<training/Adam/gradients/drop5/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_13training/Adam/gradients/drop5/cond/mul_grad/Reshape*
T0*
_class
loc:@conv5b/Relu*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ce: 

training/Adam/gradients/AddNAddN:training/Adam/gradients/drop5/cond/Switch_1_grad/cond_grad<training/Adam/gradients/drop5/cond/mul/Switch_grad/cond_grad*
T0*
_class
loc:@conv5b/Relu*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
Ă
1training/Adam/gradients/conv5b/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddNconv5b/Relu*
T0*
_class
loc:@conv5b/Relu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
Ů
7training/Adam/gradients/conv5b/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv5b/Relu_grad/ReluGrad*
T0*!
_class
loc:@conv5b/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ě
6training/Adam/gradients/conv5b/convolution_grad/ShapeNShapeNconv5a/Reluconv5b/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0*%
_class
loc:@conv5b/convolution
Š
Ctraining/Adam/gradients/conv5b/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv5b/convolution_grad/ShapeNconv5b/kernel/read1training/Adam/gradients/conv5b/Relu_grad/ReluGrad*%
_class
loc:@conv5b/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
	dilations
*
T0

Dtraining/Adam/gradients/conv5b/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv5a/Relu8training/Adam/gradients/conv5b/convolution_grad/ShapeN:11training/Adam/gradients/conv5b/Relu_grad/ReluGrad*
	dilations
*
T0*%
_class
loc:@conv5b/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:
ę
1training/Adam/gradients/conv5a/Relu_grad/ReluGradReluGradCtraining/Adam/gradients/conv5b/convolution_grad/Conv2DBackpropInputconv5a/Relu*
T0*
_class
loc:@conv5a/Relu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
Ů
7training/Adam/gradients/conv5a/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv5a/Relu_grad/ReluGrad*
T0*!
_class
loc:@conv5a/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Î
6training/Adam/gradients/conv5a/convolution_grad/ShapeNShapeNpool4/MaxPoolconv5a/kernel/read*
T0*
out_type0*%
_class
loc:@conv5a/convolution*
N* 
_output_shapes
::
Š
Ctraining/Adam/gradients/conv5a/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv5a/convolution_grad/ShapeNconv5a/kernel/read1training/Adam/gradients/conv5a/Relu_grad/ReluGrad*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
	dilations
*
T0*%
_class
loc:@conv5a/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
 
Dtraining/Adam/gradients/conv5a/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterpool4/MaxPool8training/Adam/gradients/conv5a/convolution_grad/ShapeN:11training/Adam/gradients/conv5a/Relu_grad/ReluGrad*(
_output_shapes
:*
	dilations
*
T0*%
_class
loc:@conv5a/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Ý
6training/Adam/gradients/pool4/MaxPool_grad/MaxPoolGradMaxPoolGraddrop4/cond/Mergepool4/MaxPoolCtraining/Adam/gradients/conv5a/convolution_grad/Conv2DBackpropInput*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0* 
_class
loc:@pool4/MaxPool*
strides
*
data_formatNHWC*
ksize
*
paddingVALID

training/Adam/gradients/AddN_1AddN7training/Adam/gradients/concatenate_1/concat_grad/Slice6training/Adam/gradients/pool4/MaxPool_grad/MaxPoolGrad*
T0*'
_class
loc:@concatenate_1/concat*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
ů
7training/Adam/gradients/drop4/cond/Merge_grad/cond_gradSwitchtraining/Adam/gradients/AddN_1drop4/cond/pred_id*
T0*'
_class
loc:@concatenate_1/concat*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙Ę:˙˙˙˙˙˙˙˙˙Ę
Ć
 training/Adam/gradients/Switch_2Switchconv4b/Reludrop4/cond/pred_id*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙Ę:˙˙˙˙˙˙˙˙˙Ę*
T0*
_class
loc:@conv4b/Relu
Ż
"training/Adam/gradients/Identity_2Identity"training/Adam/gradients/Switch_2:1*
T0*
_class
loc:@conv4b/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
Ą
training/Adam/gradients/Shape_3Shape"training/Adam/gradients/Switch_2:1*
T0*
out_type0*
_class
loc:@conv4b/Relu*
_output_shapes
:
Ż
%training/Adam/gradients/zeros_2/ConstConst#^training/Adam/gradients/Identity_2*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@conv4b/Relu
Ţ
training/Adam/gradients/zeros_2Filltraining/Adam/gradients/Shape_3%training/Adam/gradients/zeros_2/Const*
T0*

index_type0*
_class
loc:@conv4b/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

:training/Adam/gradients/drop4/cond/Switch_1_grad/cond_gradMerge7training/Adam/gradients/drop4/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_2*
T0*
_class
loc:@conv4b/Relu*
N*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙Ę: 
ž
9training/Adam/gradients/drop4/cond/dropout/mul_grad/ShapeShapedrop4/cond/dropout/truediv*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@drop4/cond/dropout/mul
ž
;training/Adam/gradients/drop4/cond/dropout/mul_grad/Shape_1Shapedrop4/cond/dropout/Floor*
T0*
out_type0*)
_class
loc:@drop4/cond/dropout/mul*
_output_shapes
:
Â
Itraining/Adam/gradients/drop4/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/drop4/cond/dropout/mul_grad/Shape;training/Adam/gradients/drop4/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0*)
_class
loc:@drop4/cond/dropout/mul
ű
7training/Adam/gradients/drop4/cond/dropout/mul_grad/MulMul9training/Adam/gradients/drop4/cond/Merge_grad/cond_grad:1drop4/cond/dropout/Floor*
T0*)
_class
loc:@drop4/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
­
7training/Adam/gradients/drop4/cond/dropout/mul_grad/SumSum7training/Adam/gradients/drop4/cond/dropout/mul_grad/MulItraining/Adam/gradients/drop4/cond/dropout/mul_grad/BroadcastGradientArgs*)
_class
loc:@drop4/cond/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
°
;training/Adam/gradients/drop4/cond/dropout/mul_grad/ReshapeReshape7training/Adam/gradients/drop4/cond/dropout/mul_grad/Sum9training/Adam/gradients/drop4/cond/dropout/mul_grad/Shape*
Tshape0*)
_class
loc:@drop4/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0
˙
9training/Adam/gradients/drop4/cond/dropout/mul_grad/Mul_1Muldrop4/cond/dropout/truediv9training/Adam/gradients/drop4/cond/Merge_grad/cond_grad:1*
T0*)
_class
loc:@drop4/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
ł
9training/Adam/gradients/drop4/cond/dropout/mul_grad/Sum_1Sum9training/Adam/gradients/drop4/cond/dropout/mul_grad/Mul_1Ktraining/Adam/gradients/drop4/cond/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*)
_class
loc:@drop4/cond/dropout/mul*
_output_shapes
:
ś
=training/Adam/gradients/drop4/cond/dropout/mul_grad/Reshape_1Reshape9training/Adam/gradients/drop4/cond/dropout/mul_grad/Sum_1;training/Adam/gradients/drop4/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0*
Tshape0*)
_class
loc:@drop4/cond/dropout/mul
ş
=training/Adam/gradients/drop4/cond/dropout/truediv_grad/ShapeShapedrop4/cond/mul*
out_type0*-
_class#
!loc:@drop4/cond/dropout/truediv*
_output_shapes
:*
T0
ą
?training/Adam/gradients/drop4/cond/dropout/truediv_grad/Shape_1Const*
valueB *-
_class#
!loc:@drop4/cond/dropout/truediv*
dtype0*
_output_shapes
: 
Ň
Mtraining/Adam/gradients/drop4/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/drop4/cond/dropout/truediv_grad/Shape?training/Adam/gradients/drop4/cond/dropout/truediv_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0*-
_class#
!loc:@drop4/cond/dropout/truediv

?training/Adam/gradients/drop4/cond/dropout/truediv_grad/RealDivRealDiv;training/Adam/gradients/drop4/cond/dropout/mul_grad/Reshapedrop4/cond/dropout/sub*-
_class#
!loc:@drop4/cond/dropout/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0
Á
;training/Adam/gradients/drop4/cond/dropout/truediv_grad/SumSum?training/Adam/gradients/drop4/cond/dropout/truediv_grad/RealDivMtraining/Adam/gradients/drop4/cond/dropout/truediv_grad/BroadcastGradientArgs*-
_class#
!loc:@drop4/cond/dropout/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ŕ
?training/Adam/gradients/drop4/cond/dropout/truediv_grad/ReshapeReshape;training/Adam/gradients/drop4/cond/dropout/truediv_grad/Sum=training/Adam/gradients/drop4/cond/dropout/truediv_grad/Shape*
T0*
Tshape0*-
_class#
!loc:@drop4/cond/dropout/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
ž
;training/Adam/gradients/drop4/cond/dropout/truediv_grad/NegNegdrop4/cond/mul*
T0*-
_class#
!loc:@drop4/cond/dropout/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

Atraining/Adam/gradients/drop4/cond/dropout/truediv_grad/RealDiv_1RealDiv;training/Adam/gradients/drop4/cond/dropout/truediv_grad/Negdrop4/cond/dropout/sub*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0*-
_class#
!loc:@drop4/cond/dropout/truediv

Atraining/Adam/gradients/drop4/cond/dropout/truediv_grad/RealDiv_2RealDivAtraining/Adam/gradients/drop4/cond/dropout/truediv_grad/RealDiv_1drop4/cond/dropout/sub*
T0*-
_class#
!loc:@drop4/cond/dropout/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
Ž
;training/Adam/gradients/drop4/cond/dropout/truediv_grad/mulMul;training/Adam/gradients/drop4/cond/dropout/mul_grad/ReshapeAtraining/Adam/gradients/drop4/cond/dropout/truediv_grad/RealDiv_2*
T0*-
_class#
!loc:@drop4/cond/dropout/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
Á
=training/Adam/gradients/drop4/cond/dropout/truediv_grad/Sum_1Sum;training/Adam/gradients/drop4/cond/dropout/truediv_grad/mulOtraining/Adam/gradients/drop4/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@drop4/cond/dropout/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ş
Atraining/Adam/gradients/drop4/cond/dropout/truediv_grad/Reshape_1Reshape=training/Adam/gradients/drop4/cond/dropout/truediv_grad/Sum_1?training/Adam/gradients/drop4/cond/dropout/truediv_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*-
_class#
!loc:@drop4/cond/dropout/truediv
Ť
1training/Adam/gradients/drop4/cond/mul_grad/ShapeShapedrop4/cond/mul/Switch:1*
_output_shapes
:*
T0*
out_type0*!
_class
loc:@drop4/cond/mul

3training/Adam/gradients/drop4/cond/mul_grad/Shape_1Const*
valueB *!
_class
loc:@drop4/cond/mul*
dtype0*
_output_shapes
: 
˘
Atraining/Adam/gradients/drop4/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1training/Adam/gradients/drop4/cond/mul_grad/Shape3training/Adam/gradients/drop4/cond/mul_grad/Shape_1*
T0*!
_class
loc:@drop4/cond/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
é
/training/Adam/gradients/drop4/cond/mul_grad/MulMul?training/Adam/gradients/drop4/cond/dropout/truediv_grad/Reshapedrop4/cond/mul/y*
T0*!
_class
loc:@drop4/cond/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

/training/Adam/gradients/drop4/cond/mul_grad/SumSum/training/Adam/gradients/drop4/cond/mul_grad/MulAtraining/Adam/gradients/drop4/cond/mul_grad/BroadcastGradientArgs*
T0*!
_class
loc:@drop4/cond/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 

3training/Adam/gradients/drop4/cond/mul_grad/ReshapeReshape/training/Adam/gradients/drop4/cond/mul_grad/Sum1training/Adam/gradients/drop4/cond/mul_grad/Shape*
T0*
Tshape0*!
_class
loc:@drop4/cond/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
ň
1training/Adam/gradients/drop4/cond/mul_grad/Mul_1Muldrop4/cond/mul/Switch:1?training/Adam/gradients/drop4/cond/dropout/truediv_grad/Reshape*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0*!
_class
loc:@drop4/cond/mul

1training/Adam/gradients/drop4/cond/mul_grad/Sum_1Sum1training/Adam/gradients/drop4/cond/mul_grad/Mul_1Ctraining/Adam/gradients/drop4/cond/mul_grad/BroadcastGradientArgs:1*!
_class
loc:@drop4/cond/mul*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ú
5training/Adam/gradients/drop4/cond/mul_grad/Reshape_1Reshape1training/Adam/gradients/drop4/cond/mul_grad/Sum_13training/Adam/gradients/drop4/cond/mul_grad/Shape_1*
T0*
Tshape0*!
_class
loc:@drop4/cond/mul*
_output_shapes
: 
Ć
 training/Adam/gradients/Switch_3Switchconv4b/Reludrop4/cond/pred_id*
T0*
_class
loc:@conv4b/Relu*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙Ę:˙˙˙˙˙˙˙˙˙Ę
­
"training/Adam/gradients/Identity_3Identity training/Adam/gradients/Switch_3*
T0*
_class
loc:@conv4b/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

training/Adam/gradients/Shape_4Shape training/Adam/gradients/Switch_3*
out_type0*
_class
loc:@conv4b/Relu*
_output_shapes
:*
T0
Ż
%training/Adam/gradients/zeros_3/ConstConst#^training/Adam/gradients/Identity_3*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@conv4b/Relu
Ţ
training/Adam/gradients/zeros_3Filltraining/Adam/gradients/Shape_4%training/Adam/gradients/zeros_3/Const*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0*

index_type0*
_class
loc:@conv4b/Relu

<training/Adam/gradients/drop4/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_33training/Adam/gradients/drop4/cond/mul_grad/Reshape*
N*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙Ę: *
T0*
_class
loc:@conv4b/Relu

training/Adam/gradients/AddN_2AddN:training/Adam/gradients/drop4/cond/Switch_1_grad/cond_grad<training/Adam/gradients/drop4/cond/mul/Switch_grad/cond_grad*
T0*
_class
loc:@conv4b/Relu*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
Ç
1training/Adam/gradients/conv4b/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_2conv4b/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0*
_class
loc:@conv4b/Relu
Ů
7training/Adam/gradients/conv4b/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv4b/Relu_grad/ReluGrad*
T0*!
_class
loc:@conv4b/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ě
6training/Adam/gradients/conv4b/convolution_grad/ShapeNShapeNconv4a/Reluconv4b/kernel/read*
T0*
out_type0*%
_class
loc:@conv4b/convolution*
N* 
_output_shapes
::
Ť
Ctraining/Adam/gradients/conv4b/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv4b/convolution_grad/ShapeNconv4b/kernel/read1training/Adam/gradients/conv4b/Relu_grad/ReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations
*
T0*%
_class
loc:@conv4b/convolution

Dtraining/Adam/gradients/conv4b/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv4a/Relu8training/Adam/gradients/conv4b/convolution_grad/ShapeN:11training/Adam/gradients/conv4b/Relu_grad/ReluGrad*
T0*%
_class
loc:@conv4b/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:*
	dilations

ě
1training/Adam/gradients/conv4a/Relu_grad/ReluGradReluGradCtraining/Adam/gradients/conv4b/convolution_grad/Conv2DBackpropInputconv4a/Relu*
T0*
_class
loc:@conv4a/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
Ů
7training/Adam/gradients/conv4a/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv4a/Relu_grad/ReluGrad*!
_class
loc:@conv4a/BiasAdd*
data_formatNHWC*
_output_shapes	
:*
T0
Î
6training/Adam/gradients/conv4a/convolution_grad/ShapeNShapeNpool3/MaxPoolconv4a/kernel/read*
T0*
out_type0*%
_class
loc:@conv4a/convolution*
N* 
_output_shapes
::
Ť
Ctraining/Adam/gradients/conv4a/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv4a/convolution_grad/ShapeNconv4a/kernel/read1training/Adam/gradients/conv4a/Relu_grad/ReluGrad*
	dilations
*
T0*%
_class
loc:@conv4a/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
 
Dtraining/Adam/gradients/conv4a/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterpool3/MaxPool8training/Adam/gradients/conv4a/convolution_grad/ShapeN:11training/Adam/gradients/conv4a/Relu_grad/ReluGrad*
	dilations
*
T0*%
_class
loc:@conv4a/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:
Ř
6training/Adam/gradients/pool3/MaxPool_grad/MaxPoolGradMaxPoolGradconv3b/Relupool3/MaxPoolCtraining/Adam/gradients/conv4a/convolution_grad/Conv2DBackpropInput*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0* 
_class
loc:@pool3/MaxPool*
strides
*
data_formatNHWC*
ksize
*
paddingVALID

training/Adam/gradients/AddN_3AddN7training/Adam/gradients/concatenate_2/concat_grad/Slice6training/Adam/gradients/pool3/MaxPool_grad/MaxPoolGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0*'
_class
loc:@concatenate_2/concat*
N
Ç
1training/Adam/gradients/conv3b/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_3conv3b/Relu*
T0*
_class
loc:@conv3b/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
Ů
7training/Adam/gradients/conv3b/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv3b/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0*!
_class
loc:@conv3b/BiasAdd
Ě
6training/Adam/gradients/conv3b/convolution_grad/ShapeNShapeNconv3a/Reluconv3b/kernel/read*
T0*
out_type0*%
_class
loc:@conv3b/convolution*
N* 
_output_shapes
::
Ť
Ctraining/Adam/gradients/conv3b/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv3b/convolution_grad/ShapeNconv3b/kernel/read1training/Adam/gradients/conv3b/Relu_grad/ReluGrad*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*%
_class
loc:@conv3b/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

Dtraining/Adam/gradients/conv3b/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv3a/Relu8training/Adam/gradients/conv3b/convolution_grad/ShapeN:11training/Adam/gradients/conv3b/Relu_grad/ReluGrad*(
_output_shapes
:*
	dilations
*
T0*%
_class
loc:@conv3b/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ě
1training/Adam/gradients/conv3a/Relu_grad/ReluGradReluGradCtraining/Adam/gradients/conv3b/convolution_grad/Conv2DBackpropInputconv3a/Relu*
_class
loc:@conv3a/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0
Ů
7training/Adam/gradients/conv3a/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv3a/Relu_grad/ReluGrad*
T0*!
_class
loc:@conv3a/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Î
6training/Adam/gradients/conv3a/convolution_grad/ShapeNShapeNpool2/MaxPoolconv3a/kernel/read* 
_output_shapes
::*
T0*
out_type0*%
_class
loc:@conv3a/convolution*
N
Ť
Ctraining/Adam/gradients/conv3a/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv3a/convolution_grad/ShapeNconv3a/kernel/read1training/Adam/gradients/conv3a/Relu_grad/ReluGrad*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*%
_class
loc:@conv3a/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
 
Dtraining/Adam/gradients/conv3a/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterpool2/MaxPool8training/Adam/gradients/conv3a/convolution_grad/ShapeN:11training/Adam/gradients/conv3a/Relu_grad/ReluGrad*(
_output_shapes
:*
	dilations
*
T0*%
_class
loc:@conv3a/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ř
6training/Adam/gradients/pool2/MaxPool_grad/MaxPoolGradMaxPoolGradconv2b/Relupool2/MaxPoolCtraining/Adam/gradients/conv3a/convolution_grad/Conv2DBackpropInput* 
_class
loc:@pool2/MaxPool*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0

training/Adam/gradients/AddN_4AddN7training/Adam/gradients/concatenate_3/concat_grad/Slice6training/Adam/gradients/pool2/MaxPool_grad/MaxPoolGrad*
T0*'
_class
loc:@concatenate_3/concat*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
Ç
1training/Adam/gradients/conv2b/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_4conv2b/Relu*
T0*
_class
loc:@conv2b/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
Ů
7training/Adam/gradients/conv2b/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv2b/Relu_grad/ReluGrad*
T0*!
_class
loc:@conv2b/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ě
6training/Adam/gradients/conv2b/convolution_grad/ShapeNShapeNconv2a/Reluconv2b/kernel/read*
T0*
out_type0*%
_class
loc:@conv2b/convolution*
N* 
_output_shapes
::
Ť
Ctraining/Adam/gradients/conv2b/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv2b/convolution_grad/ShapeNconv2b/kernel/read1training/Adam/gradients/conv2b/Relu_grad/ReluGrad*%
_class
loc:@conv2b/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
	dilations
*
T0

Dtraining/Adam/gradients/conv2b/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2a/Relu8training/Adam/gradients/conv2b/convolution_grad/ShapeN:11training/Adam/gradients/conv2b/Relu_grad/ReluGrad*(
_output_shapes
:*
	dilations
*
T0*%
_class
loc:@conv2b/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ě
1training/Adam/gradients/conv2a/Relu_grad/ReluGradReluGradCtraining/Adam/gradients/conv2b/convolution_grad/Conv2DBackpropInputconv2a/Relu*
T0*
_class
loc:@conv2a/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
Ů
7training/Adam/gradients/conv2a/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv2a/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0*!
_class
loc:@conv2a/BiasAdd
Î
6training/Adam/gradients/conv2a/convolution_grad/ShapeNShapeNpool1/MaxPoolconv2a/kernel/read*
T0*
out_type0*%
_class
loc:@conv2a/convolution*
N* 
_output_shapes
::
Ş
Ctraining/Adam/gradients/conv2a/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv2a/convolution_grad/ShapeNconv2a/kernel/read1training/Adam/gradients/conv2a/Relu_grad/ReluGrad*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
	dilations
*
T0*%
_class
loc:@conv2a/convolution*
data_formatNHWC*
strides


Dtraining/Adam/gradients/conv2a/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterpool1/MaxPool8training/Adam/gradients/conv2a/convolution_grad/ShapeN:11training/Adam/gradients/conv2a/Relu_grad/ReluGrad*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*%
_class
loc:@conv2a/convolution*
strides
*
data_formatNHWC
×
6training/Adam/gradients/pool1/MaxPool_grad/MaxPoolGradMaxPoolGradconv1b/Relupool1/MaxPoolCtraining/Adam/gradients/conv2a/convolution_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0* 
_class
loc:@pool1/MaxPool*
strides
*
data_formatNHWC*
ksize
*
paddingVALID

training/Adam/gradients/AddN_5AddN7training/Adam/gradients/concatenate_4/concat_grad/Slice6training/Adam/gradients/pool1/MaxPool_grad/MaxPoolGrad*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0*'
_class
loc:@concatenate_4/concat
Ć
1training/Adam/gradients/conv1b/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_5conv1b/Relu*
T0*
_class
loc:@conv1b/Relu*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
Ř
7training/Adam/gradients/conv1b/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv1b/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0*!
_class
loc:@conv1b/BiasAdd
Ě
6training/Adam/gradients/conv1b/convolution_grad/ShapeNShapeNconv1a/Reluconv1b/kernel/read* 
_output_shapes
::*
T0*
out_type0*%
_class
loc:@conv1b/convolution*
N
Ş
Ctraining/Adam/gradients/conv1b/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv1b/convolution_grad/ShapeNconv1b/kernel/read1training/Adam/gradients/conv1b/Relu_grad/ReluGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
	dilations
*
T0*%
_class
loc:@conv1b/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME

Dtraining/Adam/gradients/conv1b/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv1a/Relu8training/Adam/gradients/conv1b/convolution_grad/ShapeN:11training/Adam/gradients/conv1b/Relu_grad/ReluGrad*
paddingSAME*&
_output_shapes
:@@*
	dilations
*
T0*%
_class
loc:@conv1b/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
ë
1training/Adam/gradients/conv1a/Relu_grad/ReluGradReluGradCtraining/Adam/gradients/conv1b/convolution_grad/Conv2DBackpropInputconv1a/Relu*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0*
_class
loc:@conv1a/Relu
Ř
7training/Adam/gradients/conv1a/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv1a/Relu_grad/ReluGrad*
T0*!
_class
loc:@conv1a/BiasAdd*
data_formatNHWC*
_output_shapes
:@
Č
6training/Adam/gradients/conv1a/convolution_grad/ShapeNShapeNinput_1conv1a/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0*%
_class
loc:@conv1a/convolution
Ş
Ctraining/Adam/gradients/conv1a/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv1a/convolution_grad/ShapeNconv1a/kernel/read1training/Adam/gradients/conv1a/Relu_grad/ReluGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
	dilations
*
T0*%
_class
loc:@conv1a/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

Dtraining/Adam/gradients/conv1a/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterinput_18training/Adam/gradients/conv1a/convolution_grad/ShapeN:11training/Adam/gradients/conv1a/Relu_grad/ReluGrad*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@*
	dilations
*
T0*%
_class
loc:@conv1a/convolution*
strides
*
data_formatNHWC
_
training/Adam/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Ź
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*"
_class
loc:@Adam/iterations*
_output_shapes
: *
use_locking( *
T0	
p
training/Adam/CastCastAdam/iterations/read*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
^
training/Adam/mulMulAdam/decay/readtraining/Adam/Cast*
_output_shapes
: *
T0
X
training/Adam/add/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
a
training/Adam/addAddtraining/Adam/add/xtraining/Adam/mul*
T0*
_output_shapes
: 
\
training/Adam/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
m
training/Adam/truedivRealDivtraining/Adam/truediv/xtraining/Adam/add*
_output_shapes
: *
T0
`
training/Adam/mul_1MulAdam/lr/readtraining/Adam/truediv*
T0*
_output_shapes
: 
r
training/Adam/Cast_1CastAdam/iterations/read*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
Z
training/Adam/add_1/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/add_1Addtraining/Adam/Cast_1training/Adam/add_1/y*
T0*
_output_shapes
: 
`
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add_1*
_output_shapes
: *
T0
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
b
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add_1*
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
l
training/Adam/truediv_1RealDivtraining/Adam/Sqrttraining/Adam/sub_1*
_output_shapes
: *
T0
i
training/Adam/mul_2Multraining/Adam/mul_1training/Adam/truediv_1*
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
training/Adam/zerosFill#training/Adam/zeros/shape_as_tensortraining/Adam/zeros/Const*&
_output_shapes
:@*
T0*

index_type0

training/Adam/Variable
VariableV2*
shape:@*
shared_name *
dtype0*&
_output_shapes
:@*
	container 
Ů
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*&
_output_shapes
:@

training/Adam/Variable/readIdentitytraining/Adam/Variable*
T0*)
_class
loc:@training/Adam/Variable*&
_output_shapes
:@
b
training/Adam/zeros_1Const*
_output_shapes
:@*
valueB@*    *
dtype0
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
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:@*
use_locking(
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
training/Adam/zeros_2/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
¤
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*

index_type0*&
_output_shapes
:@@*
T0

training/Adam/Variable_2
VariableV2*
dtype0*&
_output_shapes
:@@*
	container *
shape:@@*
shared_name 
á
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2
Ą
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
dtype0*
_output_shapes
:@*
	container *
shape:@
Ő
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3
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
training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*

index_type0*'
_output_shapes
:@*
T0
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
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Ö
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
training/Adam/zeros_6/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ś
training/Adam/zeros_6Fill%training/Adam/zeros_6/shape_as_tensortraining/Adam/zeros_6/Const*(
_output_shapes
:*
T0*

index_type0
 
training/Adam/Variable_6
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ă
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*(
_output_shapes
:
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
dtype0*
_output_shapes	
:*
	container *
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
training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes	
:
~
%training/Adam/zeros_8/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"            
`
training/Adam/zeros_8/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ś
training/Adam/zeros_8Fill%training/Adam/zeros_8/shape_as_tensortraining/Adam/zeros_8/Const*
T0*

index_type0*(
_output_shapes
:
 
training/Adam/Variable_8
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ă
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Ł
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
T0*+
_class!
loc:@training/Adam/Variable_8*(
_output_shapes
:
d
training/Adam/zeros_9Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_9
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ö
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes	
:

&training/Adam/zeros_10/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_10/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Š
training/Adam/zeros_10Fill&training/Adam/zeros_10/shape_as_tensortraining/Adam/zeros_10/Const*

index_type0*(
_output_shapes
:*
T0
Ą
training/Adam/Variable_10
VariableV2*
shared_name *
dtype0*(
_output_shapes
:*
	container *
shape:
ç
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10
Ś
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_10
e
training/Adam/zeros_11Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_11
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ú
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes	
:

&training/Adam/zeros_12/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_12/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_12Fill&training/Adam/zeros_12/shape_as_tensortraining/Adam/zeros_12/Const*

index_type0*(
_output_shapes
:*
T0
Ą
training/Adam/Variable_12
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
ç
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*(
_output_shapes
:
e
training/Adam/zeros_13Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_13
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ú
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
T0*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes	
:

&training/Adam/zeros_14/shape_as_tensorConst*%
valueB"            *
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
Š
training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_14
VariableV2*(
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
ç
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*(
_output_shapes
:
e
training/Adam/zeros_15Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_15
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Ú
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_15

&training/Adam/zeros_16/shape_as_tensorConst*%
valueB"            *
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
Š
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_16
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ç
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16
Ś
training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*(
_output_shapes
:*
T0
q
&training/Adam/zeros_17/shape_as_tensorConst*
valueB:*
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

training/Adam/zeros_17Fill&training/Adam/zeros_17/shape_as_tensortraining/Adam/zeros_17/Const*
T0*

index_type0*
_output_shapes	
:

training/Adam/Variable_17
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ú
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
_output_shapes	
:

&training/Adam/zeros_18/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_18/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Š
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*

index_type0*(
_output_shapes
:*
T0
Ą
training/Adam/Variable_18
VariableV2*(
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
ç
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*
T0*,
_class"
 loc:@training/Adam/Variable_18*(
_output_shapes
:
q
&training/Adam/zeros_19/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_19/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_19Fill&training/Adam/zeros_19/shape_as_tensortraining/Adam/zeros_19/Const*
T0*

index_type0*
_output_shapes	
:

training/Adam/Variable_19
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ú
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes	
:*
use_locking(

training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_19

&training/Adam/zeros_20/shape_as_tensorConst*%
valueB"            *
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
Š
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_20
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ç
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20
Ś
training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
T0*,
_class"
 loc:@training/Adam/Variable_20*(
_output_shapes
:
e
training/Adam/zeros_21Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_21
VariableV2*
_output_shapes	
:*
	container *
shape:*
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
:

training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*
T0*,
_class"
 loc:@training/Adam/Variable_21*
_output_shapes	
:

&training/Adam/zeros_22/shape_as_tensorConst*%
valueB"            *
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
Š
training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_22
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ç
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22
Ś
training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*
T0*,
_class"
 loc:@training/Adam/Variable_22*(
_output_shapes
:
e
training/Adam/zeros_23Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_23
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ú
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*
T0*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes	
:*
use_locking(

training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*
T0*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes	
:

&training/Adam/zeros_24/shape_as_tensorConst*
_output_shapes
:*%
valueB"            *
dtype0
a
training/Adam/zeros_24/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Š
training/Adam/zeros_24Fill&training/Adam/zeros_24/shape_as_tensortraining/Adam/zeros_24/Const*

index_type0*(
_output_shapes
:*
T0
Ą
training/Adam/Variable_24
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
ç
 training/Adam/Variable_24/AssignAssigntraining/Adam/Variable_24training/Adam/zeros_24*
T0*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(*(
_output_shapes
:*
use_locking(
Ś
training/Adam/Variable_24/readIdentitytraining/Adam/Variable_24*,
_class"
 loc:@training/Adam/Variable_24*(
_output_shapes
:*
T0
e
training/Adam/zeros_25Const*
_output_shapes	
:*
valueB*    *
dtype0

training/Adam/Variable_25
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ú
 training/Adam/Variable_25/AssignAssigntraining/Adam/Variable_25training/Adam/zeros_25*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_25/readIdentitytraining/Adam/Variable_25*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_25

&training/Adam/zeros_26/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"            
a
training/Adam/zeros_26/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_26Fill&training/Adam/zeros_26/shape_as_tensortraining/Adam/zeros_26/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_26
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ç
 training/Adam/Variable_26/AssignAssigntraining/Adam/Variable_26training/Adam/zeros_26*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_26*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_26/readIdentitytraining/Adam/Variable_26*,
_class"
 loc:@training/Adam/Variable_26*(
_output_shapes
:*
T0
e
training/Adam/zeros_27Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_27
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Ú
 training/Adam/Variable_27/AssignAssigntraining/Adam/Variable_27training/Adam/zeros_27*,
_class"
 loc:@training/Adam/Variable_27*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

training/Adam/Variable_27/readIdentitytraining/Adam/Variable_27*,
_class"
 loc:@training/Adam/Variable_27*
_output_shapes	
:*
T0

&training/Adam/zeros_28/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_28/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Š
training/Adam/zeros_28Fill&training/Adam/zeros_28/shape_as_tensortraining/Adam/zeros_28/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_28
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ç
 training/Adam/Variable_28/AssignAssigntraining/Adam/Variable_28training/Adam/zeros_28*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_28*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_28/readIdentitytraining/Adam/Variable_28*
T0*,
_class"
 loc:@training/Adam/Variable_28*(
_output_shapes
:
e
training/Adam/zeros_29Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_29
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ú
 training/Adam/Variable_29/AssignAssigntraining/Adam/Variable_29training/Adam/zeros_29*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_29*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_29/readIdentitytraining/Adam/Variable_29*
T0*,
_class"
 loc:@training/Adam/Variable_29*
_output_shapes	
:

&training/Adam/zeros_30/shape_as_tensorConst*
_output_shapes
:*%
valueB"            *
dtype0
a
training/Adam/zeros_30/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Š
training/Adam/zeros_30Fill&training/Adam/zeros_30/shape_as_tensortraining/Adam/zeros_30/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_30
VariableV2*(
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
ç
 training/Adam/Variable_30/AssignAssigntraining/Adam/Variable_30training/Adam/zeros_30*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_30
Ś
training/Adam/Variable_30/readIdentitytraining/Adam/Variable_30*
T0*,
_class"
 loc:@training/Adam/Variable_30*(
_output_shapes
:
e
training/Adam/zeros_31Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_31
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Ú
 training/Adam/Variable_31/AssignAssigntraining/Adam/Variable_31training/Adam/zeros_31*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_31*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_31/readIdentitytraining/Adam/Variable_31*
T0*,
_class"
 loc:@training/Adam/Variable_31*
_output_shapes	
:

&training/Adam/zeros_32/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_32/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_32Fill&training/Adam/zeros_32/shape_as_tensortraining/Adam/zeros_32/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_32
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ç
 training/Adam/Variable_32/AssignAssigntraining/Adam/Variable_32training/Adam/zeros_32*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_32*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_32/readIdentitytraining/Adam/Variable_32*,
_class"
 loc:@training/Adam/Variable_32*(
_output_shapes
:*
T0
e
training/Adam/zeros_33Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_33
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ú
 training/Adam/Variable_33/AssignAssigntraining/Adam/Variable_33training/Adam/zeros_33*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_33*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_33/readIdentitytraining/Adam/Variable_33*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_33

&training/Adam/zeros_34/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_34/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Š
training/Adam/zeros_34Fill&training/Adam/zeros_34/shape_as_tensortraining/Adam/zeros_34/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_34
VariableV2*(
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
ç
 training/Adam/Variable_34/AssignAssigntraining/Adam/Variable_34training/Adam/zeros_34*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_34*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_34/readIdentitytraining/Adam/Variable_34*
T0*,
_class"
 loc:@training/Adam/Variable_34*(
_output_shapes
:
e
training/Adam/zeros_35Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_35
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ú
 training/Adam/Variable_35/AssignAssigntraining/Adam/Variable_35training/Adam/zeros_35*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_35*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_35/readIdentitytraining/Adam/Variable_35*,
_class"
 loc:@training/Adam/Variable_35*
_output_shapes	
:*
T0

&training/Adam/zeros_36/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_36/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_36Fill&training/Adam/zeros_36/shape_as_tensortraining/Adam/zeros_36/Const*

index_type0*(
_output_shapes
:*
T0
Ą
training/Adam/Variable_36
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
ç
 training/Adam/Variable_36/AssignAssigntraining/Adam/Variable_36training/Adam/zeros_36*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_36
Ś
training/Adam/Variable_36/readIdentitytraining/Adam/Variable_36*
T0*,
_class"
 loc:@training/Adam/Variable_36*(
_output_shapes
:
e
training/Adam/zeros_37Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_37
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Ú
 training/Adam/Variable_37/AssignAssigntraining/Adam/Variable_37training/Adam/zeros_37*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_37*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_37/readIdentitytraining/Adam/Variable_37*
T0*,
_class"
 loc:@training/Adam/Variable_37*
_output_shapes	
:

&training/Adam/zeros_38/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_38/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¨
training/Adam/zeros_38Fill&training/Adam/zeros_38/shape_as_tensortraining/Adam/zeros_38/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_38
VariableV2*
shared_name *
dtype0*'
_output_shapes
:@*
	container *
shape:@
ć
 training/Adam/Variable_38/AssignAssigntraining/Adam/Variable_38training/Adam/zeros_38*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_38*
validate_shape(*'
_output_shapes
:@
Ľ
training/Adam/Variable_38/readIdentitytraining/Adam/Variable_38*'
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_38
c
training/Adam/zeros_39Const*
_output_shapes
:@*
valueB@*    *
dtype0

training/Adam/Variable_39
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
Ů
 training/Adam/Variable_39/AssignAssigntraining/Adam/Variable_39training/Adam/zeros_39*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_39

training/Adam/Variable_39/readIdentitytraining/Adam/Variable_39*,
_class"
 loc:@training/Adam/Variable_39*
_output_shapes
:@*
T0

&training/Adam/zeros_40/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_40/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¨
training/Adam/zeros_40Fill&training/Adam/zeros_40/shape_as_tensortraining/Adam/zeros_40/Const*

index_type0*'
_output_shapes
:@*
T0

training/Adam/Variable_40
VariableV2*
shape:@*
shared_name *
dtype0*'
_output_shapes
:@*
	container 
ć
 training/Adam/Variable_40/AssignAssigntraining/Adam/Variable_40training/Adam/zeros_40*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_40
Ľ
training/Adam/Variable_40/readIdentitytraining/Adam/Variable_40*'
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_40
c
training/Adam/zeros_41Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_41
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
Ů
 training/Adam/Variable_41/AssignAssigntraining/Adam/Variable_41training/Adam/zeros_41*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_41*
validate_shape(

training/Adam/Variable_41/readIdentitytraining/Adam/Variable_41*
T0*,
_class"
 loc:@training/Adam/Variable_41*
_output_shapes
:@

&training/Adam/zeros_42/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_42/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
§
training/Adam/zeros_42Fill&training/Adam/zeros_42/shape_as_tensortraining/Adam/zeros_42/Const*
T0*

index_type0*&
_output_shapes
:@@

training/Adam/Variable_42
VariableV2*
dtype0*&
_output_shapes
:@@*
	container *
shape:@@*
shared_name 
ĺ
 training/Adam/Variable_42/AssignAssigntraining/Adam/Variable_42training/Adam/zeros_42*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_42*
validate_shape(*&
_output_shapes
:@@
¤
training/Adam/Variable_42/readIdentitytraining/Adam/Variable_42*
T0*,
_class"
 loc:@training/Adam/Variable_42*&
_output_shapes
:@@
c
training/Adam/zeros_43Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_43
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
Ů
 training/Adam/Variable_43/AssignAssigntraining/Adam/Variable_43training/Adam/zeros_43*,
_class"
 loc:@training/Adam/Variable_43*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0

training/Adam/Variable_43/readIdentitytraining/Adam/Variable_43*
T0*,
_class"
 loc:@training/Adam/Variable_43*
_output_shapes
:@
{
training/Adam/zeros_44Const*&
_output_shapes
:@*%
valueB@*    *
dtype0

training/Adam/Variable_44
VariableV2*
shared_name *
dtype0*&
_output_shapes
:@*
	container *
shape:@
ĺ
 training/Adam/Variable_44/AssignAssigntraining/Adam/Variable_44training/Adam/zeros_44*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_44*
validate_shape(*&
_output_shapes
:@
¤
training/Adam/Variable_44/readIdentitytraining/Adam/Variable_44*
T0*,
_class"
 loc:@training/Adam/Variable_44*&
_output_shapes
:@
c
training/Adam/zeros_45Const*
dtype0*
_output_shapes
:*
valueB*    

training/Adam/Variable_45
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ů
 training/Adam/Variable_45/AssignAssigntraining/Adam/Variable_45training/Adam/zeros_45*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_45*
validate_shape(*
_output_shapes
:

training/Adam/Variable_45/readIdentitytraining/Adam/Variable_45*
T0*,
_class"
 loc:@training/Adam/Variable_45*
_output_shapes
:

&training/Adam/zeros_46/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_46/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
training/Adam/zeros_46Fill&training/Adam/zeros_46/shape_as_tensortraining/Adam/zeros_46/Const*
T0*

index_type0*&
_output_shapes
:@

training/Adam/Variable_46
VariableV2*
shared_name *
dtype0*&
_output_shapes
:@*
	container *
shape:@
ĺ
 training/Adam/Variable_46/AssignAssigntraining/Adam/Variable_46training/Adam/zeros_46*&
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_46*
validate_shape(
¤
training/Adam/Variable_46/readIdentitytraining/Adam/Variable_46*
T0*,
_class"
 loc:@training/Adam/Variable_46*&
_output_shapes
:@
c
training/Adam/zeros_47Const*
dtype0*
_output_shapes
:@*
valueB@*    

training/Adam/Variable_47
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
Ů
 training/Adam/Variable_47/AssignAssigntraining/Adam/Variable_47training/Adam/zeros_47*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_47

training/Adam/Variable_47/readIdentitytraining/Adam/Variable_47*
T0*,
_class"
 loc:@training/Adam/Variable_47*
_output_shapes
:@

&training/Adam/zeros_48/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_48/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
§
training/Adam/zeros_48Fill&training/Adam/zeros_48/shape_as_tensortraining/Adam/zeros_48/Const*
T0*

index_type0*&
_output_shapes
:@@

training/Adam/Variable_48
VariableV2*&
_output_shapes
:@@*
	container *
shape:@@*
shared_name *
dtype0
ĺ
 training/Adam/Variable_48/AssignAssigntraining/Adam/Variable_48training/Adam/zeros_48*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_48*
validate_shape(*&
_output_shapes
:@@
¤
training/Adam/Variable_48/readIdentitytraining/Adam/Variable_48*
T0*,
_class"
 loc:@training/Adam/Variable_48*&
_output_shapes
:@@
c
training/Adam/zeros_49Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_49
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
Ů
 training/Adam/Variable_49/AssignAssigntraining/Adam/Variable_49training/Adam/zeros_49*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_49*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_49/readIdentitytraining/Adam/Variable_49*,
_class"
 loc:@training/Adam/Variable_49*
_output_shapes
:@*
T0

&training/Adam/zeros_50/shape_as_tensorConst*%
valueB"      @      *
dtype0*
_output_shapes
:
a
training/Adam/zeros_50/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¨
training/Adam/zeros_50Fill&training/Adam/zeros_50/shape_as_tensortraining/Adam/zeros_50/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_50
VariableV2*
dtype0*'
_output_shapes
:@*
	container *
shape:@*
shared_name 
ć
 training/Adam/Variable_50/AssignAssigntraining/Adam/Variable_50training/Adam/zeros_50*,
_class"
 loc:@training/Adam/Variable_50*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0
Ľ
training/Adam/Variable_50/readIdentitytraining/Adam/Variable_50*'
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_50
e
training/Adam/zeros_51Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_51
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ú
 training/Adam/Variable_51/AssignAssigntraining/Adam/Variable_51training/Adam/zeros_51*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_51*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_51/readIdentitytraining/Adam/Variable_51*
T0*,
_class"
 loc:@training/Adam/Variable_51*
_output_shapes	
:

&training/Adam/zeros_52/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_52/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_52Fill&training/Adam/zeros_52/shape_as_tensortraining/Adam/zeros_52/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_52
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ç
 training/Adam/Variable_52/AssignAssigntraining/Adam/Variable_52training/Adam/zeros_52*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_52*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_52/readIdentitytraining/Adam/Variable_52*
T0*,
_class"
 loc:@training/Adam/Variable_52*(
_output_shapes
:
e
training/Adam/zeros_53Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_53
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ú
 training/Adam/Variable_53/AssignAssigntraining/Adam/Variable_53training/Adam/zeros_53*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_53*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_53/readIdentitytraining/Adam/Variable_53*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_53

&training/Adam/zeros_54/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"            
a
training/Adam/zeros_54/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_54Fill&training/Adam/zeros_54/shape_as_tensortraining/Adam/zeros_54/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_54
VariableV2*
shared_name *
dtype0*(
_output_shapes
:*
	container *
shape:
ç
 training/Adam/Variable_54/AssignAssigntraining/Adam/Variable_54training/Adam/zeros_54*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_54*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_54/readIdentitytraining/Adam/Variable_54*
T0*,
_class"
 loc:@training/Adam/Variable_54*(
_output_shapes
:
e
training/Adam/zeros_55Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_55
VariableV2*
_output_shapes	
:*
	container *
shape:*
shared_name *
dtype0
Ú
 training/Adam/Variable_55/AssignAssigntraining/Adam/Variable_55training/Adam/zeros_55*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_55*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_55/readIdentitytraining/Adam/Variable_55*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_55

&training/Adam/zeros_56/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_56/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_56Fill&training/Adam/zeros_56/shape_as_tensortraining/Adam/zeros_56/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_56
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ç
 training/Adam/Variable_56/AssignAssigntraining/Adam/Variable_56training/Adam/zeros_56*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_56*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_56/readIdentitytraining/Adam/Variable_56*
T0*,
_class"
 loc:@training/Adam/Variable_56*(
_output_shapes
:
e
training/Adam/zeros_57Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_57
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ú
 training/Adam/Variable_57/AssignAssigntraining/Adam/Variable_57training/Adam/zeros_57*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_57*
validate_shape(

training/Adam/Variable_57/readIdentitytraining/Adam/Variable_57*,
_class"
 loc:@training/Adam/Variable_57*
_output_shapes	
:*
T0

&training/Adam/zeros_58/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_58/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Š
training/Adam/zeros_58Fill&training/Adam/zeros_58/shape_as_tensortraining/Adam/zeros_58/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_58
VariableV2*(
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
ç
 training/Adam/Variable_58/AssignAssigntraining/Adam/Variable_58training/Adam/zeros_58*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_58
Ś
training/Adam/Variable_58/readIdentitytraining/Adam/Variable_58*,
_class"
 loc:@training/Adam/Variable_58*(
_output_shapes
:*
T0
e
training/Adam/zeros_59Const*
_output_shapes	
:*
valueB*    *
dtype0

training/Adam/Variable_59
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Ú
 training/Adam/Variable_59/AssignAssigntraining/Adam/Variable_59training/Adam/zeros_59*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_59

training/Adam/Variable_59/readIdentitytraining/Adam/Variable_59*
T0*,
_class"
 loc:@training/Adam/Variable_59*
_output_shapes	
:

&training/Adam/zeros_60/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_60/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Š
training/Adam/zeros_60Fill&training/Adam/zeros_60/shape_as_tensortraining/Adam/zeros_60/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_60
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ç
 training/Adam/Variable_60/AssignAssigntraining/Adam/Variable_60training/Adam/zeros_60*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_60
Ś
training/Adam/Variable_60/readIdentitytraining/Adam/Variable_60*,
_class"
 loc:@training/Adam/Variable_60*(
_output_shapes
:*
T0
e
training/Adam/zeros_61Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_61
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ú
 training/Adam/Variable_61/AssignAssigntraining/Adam/Variable_61training/Adam/zeros_61*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_61*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_61/readIdentitytraining/Adam/Variable_61*,
_class"
 loc:@training/Adam/Variable_61*
_output_shapes	
:*
T0

&training/Adam/zeros_62/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_62/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_62Fill&training/Adam/zeros_62/shape_as_tensortraining/Adam/zeros_62/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_62
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ç
 training/Adam/Variable_62/AssignAssigntraining/Adam/Variable_62training/Adam/zeros_62*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_62
Ś
training/Adam/Variable_62/readIdentitytraining/Adam/Variable_62*
T0*,
_class"
 loc:@training/Adam/Variable_62*(
_output_shapes
:
q
&training/Adam/zeros_63/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_63/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_63Fill&training/Adam/zeros_63/shape_as_tensortraining/Adam/zeros_63/Const*
_output_shapes	
:*
T0*

index_type0

training/Adam/Variable_63
VariableV2*
_output_shapes	
:*
	container *
shape:*
shared_name *
dtype0
Ú
 training/Adam/Variable_63/AssignAssigntraining/Adam/Variable_63training/Adam/zeros_63*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_63

training/Adam/Variable_63/readIdentitytraining/Adam/Variable_63*
T0*,
_class"
 loc:@training/Adam/Variable_63*
_output_shapes	
:

&training/Adam/zeros_64/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_64/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_64Fill&training/Adam/zeros_64/shape_as_tensortraining/Adam/zeros_64/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_64
VariableV2*(
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
ç
 training/Adam/Variable_64/AssignAssigntraining/Adam/Variable_64training/Adam/zeros_64*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_64*
validate_shape(
Ś
training/Adam/Variable_64/readIdentitytraining/Adam/Variable_64*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_64
q
&training/Adam/zeros_65/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_65/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_65Fill&training/Adam/zeros_65/shape_as_tensortraining/Adam/zeros_65/Const*
T0*

index_type0*
_output_shapes	
:

training/Adam/Variable_65
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ú
 training/Adam/Variable_65/AssignAssigntraining/Adam/Variable_65training/Adam/zeros_65*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_65*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_65/readIdentitytraining/Adam/Variable_65*
T0*,
_class"
 loc:@training/Adam/Variable_65*
_output_shapes	
:

&training/Adam/zeros_66/shape_as_tensorConst*
_output_shapes
:*%
valueB"            *
dtype0
a
training/Adam/zeros_66/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_66Fill&training/Adam/zeros_66/shape_as_tensortraining/Adam/zeros_66/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_66
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ç
 training/Adam/Variable_66/AssignAssigntraining/Adam/Variable_66training/Adam/zeros_66*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_66*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_66/readIdentitytraining/Adam/Variable_66*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_66
e
training/Adam/zeros_67Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_67
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ú
 training/Adam/Variable_67/AssignAssigntraining/Adam/Variable_67training/Adam/zeros_67*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_67*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_67/readIdentitytraining/Adam/Variable_67*
T0*,
_class"
 loc:@training/Adam/Variable_67*
_output_shapes	
:

&training/Adam/zeros_68/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"            
a
training/Adam/zeros_68/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_68Fill&training/Adam/zeros_68/shape_as_tensortraining/Adam/zeros_68/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_68
VariableV2*(
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
ç
 training/Adam/Variable_68/AssignAssigntraining/Adam/Variable_68training/Adam/zeros_68*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_68*
validate_shape(
Ś
training/Adam/Variable_68/readIdentitytraining/Adam/Variable_68*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_68
e
training/Adam/zeros_69Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_69
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ú
 training/Adam/Variable_69/AssignAssigntraining/Adam/Variable_69training/Adam/zeros_69*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_69*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_69/readIdentitytraining/Adam/Variable_69*
T0*,
_class"
 loc:@training/Adam/Variable_69*
_output_shapes	
:

&training/Adam/zeros_70/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_70/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_70Fill&training/Adam/zeros_70/shape_as_tensortraining/Adam/zeros_70/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_70
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
ç
 training/Adam/Variable_70/AssignAssigntraining/Adam/Variable_70training/Adam/zeros_70*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_70*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_70/readIdentitytraining/Adam/Variable_70*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_70
e
training/Adam/zeros_71Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_71
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ú
 training/Adam/Variable_71/AssignAssigntraining/Adam/Variable_71training/Adam/zeros_71*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_71*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_71/readIdentitytraining/Adam/Variable_71*
T0*,
_class"
 loc:@training/Adam/Variable_71*
_output_shapes	
:

&training/Adam/zeros_72/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_72/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_72Fill&training/Adam/zeros_72/shape_as_tensortraining/Adam/zeros_72/Const*

index_type0*(
_output_shapes
:*
T0
Ą
training/Adam/Variable_72
VariableV2*
shared_name *
dtype0*(
_output_shapes
:*
	container *
shape:
ç
 training/Adam/Variable_72/AssignAssigntraining/Adam/Variable_72training/Adam/zeros_72*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_72*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_72/readIdentitytraining/Adam/Variable_72*
T0*,
_class"
 loc:@training/Adam/Variable_72*(
_output_shapes
:
e
training/Adam/zeros_73Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_73
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ú
 training/Adam/Variable_73/AssignAssigntraining/Adam/Variable_73training/Adam/zeros_73*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_73*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_73/readIdentitytraining/Adam/Variable_73*,
_class"
 loc:@training/Adam/Variable_73*
_output_shapes	
:*
T0

&training/Adam/zeros_74/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_74/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Š
training/Adam/zeros_74Fill&training/Adam/zeros_74/shape_as_tensortraining/Adam/zeros_74/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_74
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ç
 training/Adam/Variable_74/AssignAssigntraining/Adam/Variable_74training/Adam/zeros_74*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_74*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_74/readIdentitytraining/Adam/Variable_74*
T0*,
_class"
 loc:@training/Adam/Variable_74*(
_output_shapes
:
e
training/Adam/zeros_75Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_75
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ú
 training/Adam/Variable_75/AssignAssigntraining/Adam/Variable_75training/Adam/zeros_75*
T0*,
_class"
 loc:@training/Adam/Variable_75*
validate_shape(*
_output_shapes	
:*
use_locking(

training/Adam/Variable_75/readIdentitytraining/Adam/Variable_75*
T0*,
_class"
 loc:@training/Adam/Variable_75*
_output_shapes	
:

&training/Adam/zeros_76/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_76/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_76Fill&training/Adam/zeros_76/shape_as_tensortraining/Adam/zeros_76/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_76
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
ç
 training/Adam/Variable_76/AssignAssigntraining/Adam/Variable_76training/Adam/zeros_76*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_76*
validate_shape(
Ś
training/Adam/Variable_76/readIdentitytraining/Adam/Variable_76*
T0*,
_class"
 loc:@training/Adam/Variable_76*(
_output_shapes
:
e
training/Adam/zeros_77Const*
_output_shapes	
:*
valueB*    *
dtype0

training/Adam/Variable_77
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ú
 training/Adam/Variable_77/AssignAssigntraining/Adam/Variable_77training/Adam/zeros_77*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_77*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_77/readIdentitytraining/Adam/Variable_77*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_77

&training/Adam/zeros_78/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_78/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_78Fill&training/Adam/zeros_78/shape_as_tensortraining/Adam/zeros_78/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_78
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
ç
 training/Adam/Variable_78/AssignAssigntraining/Adam/Variable_78training/Adam/zeros_78*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_78*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_78/readIdentitytraining/Adam/Variable_78*
T0*,
_class"
 loc:@training/Adam/Variable_78*(
_output_shapes
:
e
training/Adam/zeros_79Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_79
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Ú
 training/Adam/Variable_79/AssignAssigntraining/Adam/Variable_79training/Adam/zeros_79*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_79*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_79/readIdentitytraining/Adam/Variable_79*
T0*,
_class"
 loc:@training/Adam/Variable_79*
_output_shapes	
:

&training/Adam/zeros_80/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_80/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_80Fill&training/Adam/zeros_80/shape_as_tensortraining/Adam/zeros_80/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_80
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ç
 training/Adam/Variable_80/AssignAssigntraining/Adam/Variable_80training/Adam/zeros_80*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_80*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_80/readIdentitytraining/Adam/Variable_80*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_80
e
training/Adam/zeros_81Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_81
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ú
 training/Adam/Variable_81/AssignAssigntraining/Adam/Variable_81training/Adam/zeros_81*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_81*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_81/readIdentitytraining/Adam/Variable_81*
T0*,
_class"
 loc:@training/Adam/Variable_81*
_output_shapes	
:

&training/Adam/zeros_82/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_82/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Š
training/Adam/zeros_82Fill&training/Adam/zeros_82/shape_as_tensortraining/Adam/zeros_82/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_82
VariableV2*
shared_name *
dtype0*(
_output_shapes
:*
	container *
shape:
ç
 training/Adam/Variable_82/AssignAssigntraining/Adam/Variable_82training/Adam/zeros_82*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_82*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_82/readIdentitytraining/Adam/Variable_82*
T0*,
_class"
 loc:@training/Adam/Variable_82*(
_output_shapes
:
e
training/Adam/zeros_83Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_83
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ú
 training/Adam/Variable_83/AssignAssigntraining/Adam/Variable_83training/Adam/zeros_83*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_83*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_83/readIdentitytraining/Adam/Variable_83*
T0*,
_class"
 loc:@training/Adam/Variable_83*
_output_shapes	
:

&training/Adam/zeros_84/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_84/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¨
training/Adam/zeros_84Fill&training/Adam/zeros_84/shape_as_tensortraining/Adam/zeros_84/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_84
VariableV2*'
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
ć
 training/Adam/Variable_84/AssignAssigntraining/Adam/Variable_84training/Adam/zeros_84*'
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_84*
validate_shape(
Ľ
training/Adam/Variable_84/readIdentitytraining/Adam/Variable_84*
T0*,
_class"
 loc:@training/Adam/Variable_84*'
_output_shapes
:@
c
training/Adam/zeros_85Const*
_output_shapes
:@*
valueB@*    *
dtype0

training/Adam/Variable_85
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
Ů
 training/Adam/Variable_85/AssignAssigntraining/Adam/Variable_85training/Adam/zeros_85*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_85*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_85/readIdentitytraining/Adam/Variable_85*
T0*,
_class"
 loc:@training/Adam/Variable_85*
_output_shapes
:@

&training/Adam/zeros_86/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_86/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
¨
training/Adam/zeros_86Fill&training/Adam/zeros_86/shape_as_tensortraining/Adam/zeros_86/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_86
VariableV2*
shared_name *
dtype0*'
_output_shapes
:@*
	container *
shape:@
ć
 training/Adam/Variable_86/AssignAssigntraining/Adam/Variable_86training/Adam/zeros_86*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_86
Ľ
training/Adam/Variable_86/readIdentitytraining/Adam/Variable_86*
T0*,
_class"
 loc:@training/Adam/Variable_86*'
_output_shapes
:@
c
training/Adam/zeros_87Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_87
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
Ů
 training/Adam/Variable_87/AssignAssigntraining/Adam/Variable_87training/Adam/zeros_87*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_87*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_87/readIdentitytraining/Adam/Variable_87*
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_87

&training/Adam/zeros_88/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_88/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
training/Adam/zeros_88Fill&training/Adam/zeros_88/shape_as_tensortraining/Adam/zeros_88/Const*
T0*

index_type0*&
_output_shapes
:@@

training/Adam/Variable_88
VariableV2*
shared_name *
dtype0*&
_output_shapes
:@@*
	container *
shape:@@
ĺ
 training/Adam/Variable_88/AssignAssigntraining/Adam/Variable_88training/Adam/zeros_88*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_88*
validate_shape(*&
_output_shapes
:@@
¤
training/Adam/Variable_88/readIdentitytraining/Adam/Variable_88*
T0*,
_class"
 loc:@training/Adam/Variable_88*&
_output_shapes
:@@
c
training/Adam/zeros_89Const*
_output_shapes
:@*
valueB@*    *
dtype0

training/Adam/Variable_89
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
Ů
 training/Adam/Variable_89/AssignAssigntraining/Adam/Variable_89training/Adam/zeros_89*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_89*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_89/readIdentitytraining/Adam/Variable_89*
T0*,
_class"
 loc:@training/Adam/Variable_89*
_output_shapes
:@
{
training/Adam/zeros_90Const*%
valueB@*    *
dtype0*&
_output_shapes
:@

training/Adam/Variable_90
VariableV2*
shape:@*
shared_name *
dtype0*&
_output_shapes
:@*
	container 
ĺ
 training/Adam/Variable_90/AssignAssigntraining/Adam/Variable_90training/Adam/zeros_90*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_90*
validate_shape(*&
_output_shapes
:@
¤
training/Adam/Variable_90/readIdentitytraining/Adam/Variable_90*&
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_90
c
training/Adam/zeros_91Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_91
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
Ů
 training/Adam/Variable_91/AssignAssigntraining/Adam/Variable_91training/Adam/zeros_91*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_91*
validate_shape(

training/Adam/Variable_91/readIdentitytraining/Adam/Variable_91*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_91
p
&training/Adam/zeros_92/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_92/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_92Fill&training/Adam/zeros_92/shape_as_tensortraining/Adam/zeros_92/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_92
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ů
 training/Adam/Variable_92/AssignAssigntraining/Adam/Variable_92training/Adam/zeros_92*
T0*,
_class"
 loc:@training/Adam/Variable_92*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_92/readIdentitytraining/Adam/Variable_92*
T0*,
_class"
 loc:@training/Adam/Variable_92*
_output_shapes
:
p
&training/Adam/zeros_93/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_93/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_93Fill&training/Adam/zeros_93/shape_as_tensortraining/Adam/zeros_93/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_93
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ů
 training/Adam/Variable_93/AssignAssigntraining/Adam/Variable_93training/Adam/zeros_93*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_93

training/Adam/Variable_93/readIdentitytraining/Adam/Variable_93*
T0*,
_class"
 loc:@training/Adam/Variable_93*
_output_shapes
:
p
&training/Adam/zeros_94/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_94/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_94Fill&training/Adam/zeros_94/shape_as_tensortraining/Adam/zeros_94/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_94
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
Ů
 training/Adam/Variable_94/AssignAssigntraining/Adam/Variable_94training/Adam/zeros_94*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_94*
validate_shape(*
_output_shapes
:

training/Adam/Variable_94/readIdentitytraining/Adam/Variable_94*
T0*,
_class"
 loc:@training/Adam/Variable_94*
_output_shapes
:
p
&training/Adam/zeros_95/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_95/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_95Fill&training/Adam/zeros_95/shape_as_tensortraining/Adam/zeros_95/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_95
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ů
 training/Adam/Variable_95/AssignAssigntraining/Adam/Variable_95training/Adam/zeros_95*,
_class"
 loc:@training/Adam/Variable_95*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_95/readIdentitytraining/Adam/Variable_95*
T0*,
_class"
 loc:@training/Adam/Variable_95*
_output_shapes
:
p
&training/Adam/zeros_96/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_96/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_96Fill&training/Adam/zeros_96/shape_as_tensortraining/Adam/zeros_96/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_96
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ů
 training/Adam/Variable_96/AssignAssigntraining/Adam/Variable_96training/Adam/zeros_96*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_96*
validate_shape(

training/Adam/Variable_96/readIdentitytraining/Adam/Variable_96*,
_class"
 loc:@training/Adam/Variable_96*
_output_shapes
:*
T0
p
&training/Adam/zeros_97/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_97/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_97Fill&training/Adam/zeros_97/shape_as_tensortraining/Adam/zeros_97/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_97
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ů
 training/Adam/Variable_97/AssignAssigntraining/Adam/Variable_97training/Adam/zeros_97*,
_class"
 loc:@training/Adam/Variable_97*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_97/readIdentitytraining/Adam/Variable_97*
T0*,
_class"
 loc:@training/Adam/Variable_97*
_output_shapes
:
p
&training/Adam/zeros_98/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_98/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_98Fill&training/Adam/zeros_98/shape_as_tensortraining/Adam/zeros_98/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_98
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ů
 training/Adam/Variable_98/AssignAssigntraining/Adam/Variable_98training/Adam/zeros_98*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_98*
validate_shape(

training/Adam/Variable_98/readIdentitytraining/Adam/Variable_98*
T0*,
_class"
 loc:@training/Adam/Variable_98*
_output_shapes
:
p
&training/Adam/zeros_99/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_99/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_99Fill&training/Adam/zeros_99/shape_as_tensortraining/Adam/zeros_99/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_99
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ů
 training/Adam/Variable_99/AssignAssigntraining/Adam/Variable_99training/Adam/zeros_99*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_99*
validate_shape(*
_output_shapes
:

training/Adam/Variable_99/readIdentitytraining/Adam/Variable_99*,
_class"
 loc:@training/Adam/Variable_99*
_output_shapes
:*
T0
q
'training/Adam/zeros_100/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_100/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_100Fill'training/Adam/zeros_100/shape_as_tensortraining/Adam/zeros_100/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_100
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
Ý
!training/Adam/Variable_100/AssignAssigntraining/Adam/Variable_100training/Adam/zeros_100*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_100*
validate_shape(*
_output_shapes
:

training/Adam/Variable_100/readIdentitytraining/Adam/Variable_100*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_100
q
'training/Adam/zeros_101/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_101/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_101Fill'training/Adam/zeros_101/shape_as_tensortraining/Adam/zeros_101/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_101
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ý
!training/Adam/Variable_101/AssignAssigntraining/Adam/Variable_101training/Adam/zeros_101*-
_class#
!loc:@training/Adam/Variable_101*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_101/readIdentitytraining/Adam/Variable_101*
T0*-
_class#
!loc:@training/Adam/Variable_101*
_output_shapes
:
q
'training/Adam/zeros_102/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_102/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_102Fill'training/Adam/zeros_102/shape_as_tensortraining/Adam/zeros_102/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_102
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_102/AssignAssigntraining/Adam/Variable_102training/Adam/zeros_102*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_102*
validate_shape(

training/Adam/Variable_102/readIdentitytraining/Adam/Variable_102*
T0*-
_class#
!loc:@training/Adam/Variable_102*
_output_shapes
:
q
'training/Adam/zeros_103/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_103/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_103Fill'training/Adam/zeros_103/shape_as_tensortraining/Adam/zeros_103/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_103
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ý
!training/Adam/Variable_103/AssignAssigntraining/Adam/Variable_103training/Adam/zeros_103*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_103*
validate_shape(*
_output_shapes
:

training/Adam/Variable_103/readIdentitytraining/Adam/Variable_103*
T0*-
_class#
!loc:@training/Adam/Variable_103*
_output_shapes
:
q
'training/Adam/zeros_104/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_104/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_104Fill'training/Adam/zeros_104/shape_as_tensortraining/Adam/zeros_104/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_104
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_104/AssignAssigntraining/Adam/Variable_104training/Adam/zeros_104*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_104*
validate_shape(

training/Adam/Variable_104/readIdentitytraining/Adam/Variable_104*
T0*-
_class#
!loc:@training/Adam/Variable_104*
_output_shapes
:
q
'training/Adam/zeros_105/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
b
training/Adam/zeros_105/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_105Fill'training/Adam/zeros_105/shape_as_tensortraining/Adam/zeros_105/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_105
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ý
!training/Adam/Variable_105/AssignAssigntraining/Adam/Variable_105training/Adam/zeros_105*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_105*
validate_shape(*
_output_shapes
:

training/Adam/Variable_105/readIdentitytraining/Adam/Variable_105*
T0*-
_class#
!loc:@training/Adam/Variable_105*
_output_shapes
:
q
'training/Adam/zeros_106/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_106/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_106Fill'training/Adam/zeros_106/shape_as_tensortraining/Adam/zeros_106/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_106
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_106/AssignAssigntraining/Adam/Variable_106training/Adam/zeros_106*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_106*
validate_shape(*
_output_shapes
:

training/Adam/Variable_106/readIdentitytraining/Adam/Variable_106*
T0*-
_class#
!loc:@training/Adam/Variable_106*
_output_shapes
:
q
'training/Adam/zeros_107/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_107/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_107Fill'training/Adam/zeros_107/shape_as_tensortraining/Adam/zeros_107/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_107
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
Ý
!training/Adam/Variable_107/AssignAssigntraining/Adam/Variable_107training/Adam/zeros_107*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_107*
validate_shape(*
_output_shapes
:

training/Adam/Variable_107/readIdentitytraining/Adam/Variable_107*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_107
q
'training/Adam/zeros_108/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
b
training/Adam/zeros_108/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_108Fill'training/Adam/zeros_108/shape_as_tensortraining/Adam/zeros_108/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_108
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ý
!training/Adam/Variable_108/AssignAssigntraining/Adam/Variable_108training/Adam/zeros_108*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_108

training/Adam/Variable_108/readIdentitytraining/Adam/Variable_108*
T0*-
_class#
!loc:@training/Adam/Variable_108*
_output_shapes
:
q
'training/Adam/zeros_109/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_109/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_109Fill'training/Adam/zeros_109/shape_as_tensortraining/Adam/zeros_109/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_109
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ý
!training/Adam/Variable_109/AssignAssigntraining/Adam/Variable_109training/Adam/zeros_109*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_109*
validate_shape(*
_output_shapes
:

training/Adam/Variable_109/readIdentitytraining/Adam/Variable_109*-
_class#
!loc:@training/Adam/Variable_109*
_output_shapes
:*
T0
q
'training/Adam/zeros_110/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
b
training/Adam/zeros_110/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_110Fill'training/Adam/zeros_110/shape_as_tensortraining/Adam/zeros_110/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_110
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ý
!training/Adam/Variable_110/AssignAssigntraining/Adam/Variable_110training/Adam/zeros_110*
T0*-
_class#
!loc:@training/Adam/Variable_110*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_110/readIdentitytraining/Adam/Variable_110*
T0*-
_class#
!loc:@training/Adam/Variable_110*
_output_shapes
:
q
'training/Adam/zeros_111/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_111/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_111Fill'training/Adam/zeros_111/shape_as_tensortraining/Adam/zeros_111/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_111
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ý
!training/Adam/Variable_111/AssignAssigntraining/Adam/Variable_111training/Adam/zeros_111*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_111

training/Adam/Variable_111/readIdentitytraining/Adam/Variable_111*
T0*-
_class#
!loc:@training/Adam/Variable_111*
_output_shapes
:
q
'training/Adam/zeros_112/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_112/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_112Fill'training/Adam/zeros_112/shape_as_tensortraining/Adam/zeros_112/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_112
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_112/AssignAssigntraining/Adam/Variable_112training/Adam/zeros_112*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_112*
validate_shape(*
_output_shapes
:

training/Adam/Variable_112/readIdentitytraining/Adam/Variable_112*
T0*-
_class#
!loc:@training/Adam/Variable_112*
_output_shapes
:
q
'training/Adam/zeros_113/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
b
training/Adam/zeros_113/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_113Fill'training/Adam/zeros_113/shape_as_tensortraining/Adam/zeros_113/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_113
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ý
!training/Adam/Variable_113/AssignAssigntraining/Adam/Variable_113training/Adam/zeros_113*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_113*
validate_shape(

training/Adam/Variable_113/readIdentitytraining/Adam/Variable_113*
T0*-
_class#
!loc:@training/Adam/Variable_113*
_output_shapes
:
q
'training/Adam/zeros_114/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_114/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_114Fill'training/Adam/zeros_114/shape_as_tensortraining/Adam/zeros_114/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_114
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ý
!training/Adam/Variable_114/AssignAssigntraining/Adam/Variable_114training/Adam/zeros_114*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_114*
validate_shape(*
_output_shapes
:

training/Adam/Variable_114/readIdentitytraining/Adam/Variable_114*
T0*-
_class#
!loc:@training/Adam/Variable_114*
_output_shapes
:
q
'training/Adam/zeros_115/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_115/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_115Fill'training/Adam/zeros_115/shape_as_tensortraining/Adam/zeros_115/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_115
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ý
!training/Adam/Variable_115/AssignAssigntraining/Adam/Variable_115training/Adam/zeros_115*-
_class#
!loc:@training/Adam/Variable_115*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_115/readIdentitytraining/Adam/Variable_115*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_115
q
'training/Adam/zeros_116/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
b
training/Adam/zeros_116/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_116Fill'training/Adam/zeros_116/shape_as_tensortraining/Adam/zeros_116/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_116
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_116/AssignAssigntraining/Adam/Variable_116training/Adam/zeros_116*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_116*
validate_shape(*
_output_shapes
:

training/Adam/Variable_116/readIdentitytraining/Adam/Variable_116*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_116
q
'training/Adam/zeros_117/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_117/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_117Fill'training/Adam/zeros_117/shape_as_tensortraining/Adam/zeros_117/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_117
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_117/AssignAssigntraining/Adam/Variable_117training/Adam/zeros_117*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_117*
validate_shape(*
_output_shapes
:

training/Adam/Variable_117/readIdentitytraining/Adam/Variable_117*-
_class#
!loc:@training/Adam/Variable_117*
_output_shapes
:*
T0
q
'training/Adam/zeros_118/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_118/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_118Fill'training/Adam/zeros_118/shape_as_tensortraining/Adam/zeros_118/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_118
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_118/AssignAssigntraining/Adam/Variable_118training/Adam/zeros_118*-
_class#
!loc:@training/Adam/Variable_118*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_118/readIdentitytraining/Adam/Variable_118*
T0*-
_class#
!loc:@training/Adam/Variable_118*
_output_shapes
:
q
'training/Adam/zeros_119/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
b
training/Adam/zeros_119/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_119Fill'training/Adam/zeros_119/shape_as_tensortraining/Adam/zeros_119/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_119
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
Ý
!training/Adam/Variable_119/AssignAssigntraining/Adam/Variable_119training/Adam/zeros_119*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_119*
validate_shape(*
_output_shapes
:

training/Adam/Variable_119/readIdentitytraining/Adam/Variable_119*
T0*-
_class#
!loc:@training/Adam/Variable_119*
_output_shapes
:
q
'training/Adam/zeros_120/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_120/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_120Fill'training/Adam/zeros_120/shape_as_tensortraining/Adam/zeros_120/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_120
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ý
!training/Adam/Variable_120/AssignAssigntraining/Adam/Variable_120training/Adam/zeros_120*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_120*
validate_shape(

training/Adam/Variable_120/readIdentitytraining/Adam/Variable_120*
T0*-
_class#
!loc:@training/Adam/Variable_120*
_output_shapes
:
q
'training/Adam/zeros_121/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_121/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_121Fill'training/Adam/zeros_121/shape_as_tensortraining/Adam/zeros_121/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_121
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ý
!training/Adam/Variable_121/AssignAssigntraining/Adam/Variable_121training/Adam/zeros_121*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_121*
validate_shape(*
_output_shapes
:

training/Adam/Variable_121/readIdentitytraining/Adam/Variable_121*
T0*-
_class#
!loc:@training/Adam/Variable_121*
_output_shapes
:
q
'training/Adam/zeros_122/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_122/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_122Fill'training/Adam/zeros_122/shape_as_tensortraining/Adam/zeros_122/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_122
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_122/AssignAssigntraining/Adam/Variable_122training/Adam/zeros_122*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_122

training/Adam/Variable_122/readIdentitytraining/Adam/Variable_122*
T0*-
_class#
!loc:@training/Adam/Variable_122*
_output_shapes
:
q
'training/Adam/zeros_123/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
b
training/Adam/zeros_123/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_123Fill'training/Adam/zeros_123/shape_as_tensortraining/Adam/zeros_123/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_123
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ý
!training/Adam/Variable_123/AssignAssigntraining/Adam/Variable_123training/Adam/zeros_123*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_123*
validate_shape(*
_output_shapes
:

training/Adam/Variable_123/readIdentitytraining/Adam/Variable_123*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_123
q
'training/Adam/zeros_124/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_124/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_124Fill'training/Adam/zeros_124/shape_as_tensortraining/Adam/zeros_124/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_124
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ý
!training/Adam/Variable_124/AssignAssigntraining/Adam/Variable_124training/Adam/zeros_124*-
_class#
!loc:@training/Adam/Variable_124*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_124/readIdentitytraining/Adam/Variable_124*
T0*-
_class#
!loc:@training/Adam/Variable_124*
_output_shapes
:
q
'training/Adam/zeros_125/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_125/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_125Fill'training/Adam/zeros_125/shape_as_tensortraining/Adam/zeros_125/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_125
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_125/AssignAssigntraining/Adam/Variable_125training/Adam/zeros_125*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_125*
validate_shape(*
_output_shapes
:

training/Adam/Variable_125/readIdentitytraining/Adam/Variable_125*-
_class#
!loc:@training/Adam/Variable_125*
_output_shapes
:*
T0
q
'training/Adam/zeros_126/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_126/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_126Fill'training/Adam/zeros_126/shape_as_tensortraining/Adam/zeros_126/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_126
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_126/AssignAssigntraining/Adam/Variable_126training/Adam/zeros_126*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_126*
validate_shape(*
_output_shapes
:

training/Adam/Variable_126/readIdentitytraining/Adam/Variable_126*
T0*-
_class#
!loc:@training/Adam/Variable_126*
_output_shapes
:
q
'training/Adam/zeros_127/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_127/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_127Fill'training/Adam/zeros_127/shape_as_tensortraining/Adam/zeros_127/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_127
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ý
!training/Adam/Variable_127/AssignAssigntraining/Adam/Variable_127training/Adam/zeros_127*
T0*-
_class#
!loc:@training/Adam/Variable_127*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_127/readIdentitytraining/Adam/Variable_127*-
_class#
!loc:@training/Adam/Variable_127*
_output_shapes
:*
T0
q
'training/Adam/zeros_128/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_128/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_128Fill'training/Adam/zeros_128/shape_as_tensortraining/Adam/zeros_128/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_128
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_128/AssignAssigntraining/Adam/Variable_128training/Adam/zeros_128*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_128*
validate_shape(*
_output_shapes
:

training/Adam/Variable_128/readIdentitytraining/Adam/Variable_128*
T0*-
_class#
!loc:@training/Adam/Variable_128*
_output_shapes
:
q
'training/Adam/zeros_129/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_129/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_129Fill'training/Adam/zeros_129/shape_as_tensortraining/Adam/zeros_129/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_129
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ý
!training/Adam/Variable_129/AssignAssigntraining/Adam/Variable_129training/Adam/zeros_129*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_129*
validate_shape(*
_output_shapes
:

training/Adam/Variable_129/readIdentitytraining/Adam/Variable_129*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_129
q
'training/Adam/zeros_130/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_130/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_130Fill'training/Adam/zeros_130/shape_as_tensortraining/Adam/zeros_130/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_130
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_130/AssignAssigntraining/Adam/Variable_130training/Adam/zeros_130*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_130*
validate_shape(*
_output_shapes
:

training/Adam/Variable_130/readIdentitytraining/Adam/Variable_130*-
_class#
!loc:@training/Adam/Variable_130*
_output_shapes
:*
T0
q
'training/Adam/zeros_131/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_131/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_131Fill'training/Adam/zeros_131/shape_as_tensortraining/Adam/zeros_131/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_131
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ý
!training/Adam/Variable_131/AssignAssigntraining/Adam/Variable_131training/Adam/zeros_131*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_131*
validate_shape(*
_output_shapes
:

training/Adam/Variable_131/readIdentitytraining/Adam/Variable_131*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_131
q
'training/Adam/zeros_132/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_132/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_132Fill'training/Adam/zeros_132/shape_as_tensortraining/Adam/zeros_132/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_132
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_132/AssignAssigntraining/Adam/Variable_132training/Adam/zeros_132*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_132*
validate_shape(*
_output_shapes
:

training/Adam/Variable_132/readIdentitytraining/Adam/Variable_132*
T0*-
_class#
!loc:@training/Adam/Variable_132*
_output_shapes
:
q
'training/Adam/zeros_133/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_133/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_133Fill'training/Adam/zeros_133/shape_as_tensortraining/Adam/zeros_133/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_133
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_133/AssignAssigntraining/Adam/Variable_133training/Adam/zeros_133*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_133*
validate_shape(

training/Adam/Variable_133/readIdentitytraining/Adam/Variable_133*-
_class#
!loc:@training/Adam/Variable_133*
_output_shapes
:*
T0
q
'training/Adam/zeros_134/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
b
training/Adam/zeros_134/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_134Fill'training/Adam/zeros_134/shape_as_tensortraining/Adam/zeros_134/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_134
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ý
!training/Adam/Variable_134/AssignAssigntraining/Adam/Variable_134training/Adam/zeros_134*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_134*
validate_shape(*
_output_shapes
:

training/Adam/Variable_134/readIdentitytraining/Adam/Variable_134*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_134
q
'training/Adam/zeros_135/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_135/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_135Fill'training/Adam/zeros_135/shape_as_tensortraining/Adam/zeros_135/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_135
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ý
!training/Adam/Variable_135/AssignAssigntraining/Adam/Variable_135training/Adam/zeros_135*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_135*
validate_shape(

training/Adam/Variable_135/readIdentitytraining/Adam/Variable_135*
T0*-
_class#
!loc:@training/Adam/Variable_135*
_output_shapes
:
q
'training/Adam/zeros_136/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
b
training/Adam/zeros_136/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_136Fill'training/Adam/zeros_136/shape_as_tensortraining/Adam/zeros_136/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_136
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_136/AssignAssigntraining/Adam/Variable_136training/Adam/zeros_136*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_136

training/Adam/Variable_136/readIdentitytraining/Adam/Variable_136*-
_class#
!loc:@training/Adam/Variable_136*
_output_shapes
:*
T0
q
'training/Adam/zeros_137/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_137/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_137Fill'training/Adam/zeros_137/shape_as_tensortraining/Adam/zeros_137/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_137
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Ý
!training/Adam/Variable_137/AssignAssigntraining/Adam/Variable_137training/Adam/zeros_137*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_137*
validate_shape(*
_output_shapes
:

training/Adam/Variable_137/readIdentitytraining/Adam/Variable_137*
T0*-
_class#
!loc:@training/Adam/Variable_137*
_output_shapes
:
z
training/Adam/mul_3MulAdam/beta_1/readtraining/Adam/Variable/read*
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
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
_output_shapes
: *
T0
Ś
training/Adam/mul_4Multraining/Adam/sub_2Dtraining/Adam/gradients/conv1a/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
u
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*&
_output_shapes
:@
}
training/Adam/mul_5MulAdam/beta_2/readtraining/Adam/Variable_46/read*
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
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/SquareSquareDtraining/Adam/gradients/conv1a/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
v
training/Adam/mul_6Multraining/Adam/sub_3training/Adam/Square*&
_output_shapes
:@*
T0
u
training/Adam/add_3Addtraining/Adam/mul_5training/Adam/mul_6*
T0*&
_output_shapes
:@
u
training/Adam/mul_7Multraining/Adam/mul_2training/Adam/add_2*
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
training/Adam/Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *  

%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_3training/Adam/Const_3*
T0*&
_output_shapes
:@

training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*&
_output_shapes
:@*
T0
l
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*&
_output_shapes
:@*
T0
Z
training/Adam/add_4/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
x
training/Adam/add_4Addtraining/Adam/Sqrt_1training/Adam/add_4/y*
T0*&
_output_shapes
:@
}
training/Adam/truediv_2RealDivtraining/Adam/mul_7training/Adam/add_4*
T0*&
_output_shapes
:@
x
training/Adam/sub_4Subconv1a/kernel/readtraining/Adam/truediv_2*
T0*&
_output_shapes
:@
Đ
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_2*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*&
_output_shapes
:@
Ř
training/Adam/Assign_1Assigntraining/Adam/Variable_46training/Adam/add_3*,
_class"
 loc:@training/Adam/Variable_46*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
Ŕ
training/Adam/Assign_2Assignconv1a/kerneltraining/Adam/sub_4* 
_class
loc:@conv1a/kernel*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
p
training/Adam/mul_8MulAdam/beta_1/readtraining/Adam/Variable_1/read*
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
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_9Multraining/Adam/sub_57training/Adam/gradients/conv1a/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
:@
r
training/Adam/mul_10MulAdam/beta_2/readtraining/Adam/Variable_47/read*
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
~
training/Adam/Square_1Square7training/Adam/gradients/conv1a/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
m
training/Adam/mul_11Multraining/Adam/sub_6training/Adam/Square_1*
_output_shapes
:@*
T0
k
training/Adam/add_6Addtraining/Adam/mul_10training/Adam/mul_11*
T0*
_output_shapes
:@
j
training/Adam/mul_12Multraining/Adam/mul_2training/Adam/add_5*
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
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_6training/Adam/Const_5*
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
training/Adam/add_7/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
l
training/Adam/add_7Addtraining/Adam/Sqrt_2training/Adam/add_7/y*
T0*
_output_shapes
:@
r
training/Adam/truediv_3RealDivtraining/Adam/mul_12training/Adam/add_7*
_output_shapes
:@*
T0
j
training/Adam/sub_7Subconv1a/bias/readtraining/Adam/truediv_3*
T0*
_output_shapes
:@
Ę
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_5*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:@
Ě
training/Adam/Assign_4Assigntraining/Adam/Variable_47training/Adam/add_6*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_47*
validate_shape(*
_output_shapes
:@
°
training/Adam/Assign_5Assignconv1a/biastraining/Adam/sub_7*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv1a/bias
}
training/Adam/mul_13MulAdam/beta_1/readtraining/Adam/Variable_2/read*&
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
§
training/Adam/mul_14Multraining/Adam/sub_8Dtraining/Adam/gradients/conv1b/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
w
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*&
_output_shapes
:@@*
T0
~
training/Adam/mul_15MulAdam/beta_2/readtraining/Adam/Variable_48/read*
T0*&
_output_shapes
:@@
Z
training/Adam/sub_9/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_2SquareDtraining/Adam/gradients/conv1b/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@@
y
training/Adam/mul_16Multraining/Adam/sub_9training/Adam/Square_2*
T0*&
_output_shapes
:@@
w
training/Adam/add_9Addtraining/Adam/mul_15training/Adam/mul_16*&
_output_shapes
:@@*
T0
v
training/Adam/mul_17Multraining/Adam/mul_2training/Adam/add_8*&
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
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_9training/Adam/Const_7*&
_output_shapes
:@@*
T0
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
[
training/Adam/add_10/yConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
z
training/Adam/add_10Addtraining/Adam/Sqrt_3training/Adam/add_10/y*&
_output_shapes
:@@*
T0

training/Adam/truediv_4RealDivtraining/Adam/mul_17training/Adam/add_10*
T0*&
_output_shapes
:@@
y
training/Adam/sub_10Subconv1b/kernel/readtraining/Adam/truediv_4*
T0*&
_output_shapes
:@@
Ö
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_8*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2
Ř
training/Adam/Assign_7Assigntraining/Adam/Variable_48training/Adam/add_9*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_48
Á
training/Adam/Assign_8Assignconv1b/kerneltraining/Adam/sub_10*&
_output_shapes
:@@*
use_locking(*
T0* 
_class
loc:@conv1b/kernel*
validate_shape(
q
training/Adam/mul_18MulAdam/beta_1/readtraining/Adam/Variable_3/read*
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

training/Adam/mul_19Multraining/Adam/sub_117training/Adam/gradients/conv1b/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
_output_shapes
:@*
T0
r
training/Adam/mul_20MulAdam/beta_2/readtraining/Adam/Variable_49/read*
_output_shapes
:@*
T0
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
~
training/Adam/Square_3Square7training/Adam/gradients/conv1b/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
n
training/Adam/mul_21Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes
:@
l
training/Adam/add_12Addtraining/Adam/mul_20training/Adam/mul_21*
T0*
_output_shapes
:@
k
training/Adam/mul_22Multraining/Adam/mul_2training/Adam/add_11*
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
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_12training/Adam/Const_9*
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
training/Adam/add_13/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
n
training/Adam/add_13Addtraining/Adam/Sqrt_4training/Adam/add_13/y*
T0*
_output_shapes
:@
s
training/Adam/truediv_5RealDivtraining/Adam/mul_22training/Adam/add_13*
T0*
_output_shapes
:@
k
training/Adam/sub_13Subconv1b/bias/readtraining/Adam/truediv_5*
_output_shapes
:@*
T0
Ë
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_11*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:@
Î
training/Adam/Assign_10Assigntraining/Adam/Variable_49training/Adam/add_12*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_49*
validate_shape(*
_output_shapes
:@
˛
training/Adam/Assign_11Assignconv1b/biastraining/Adam/sub_13*
use_locking(*
T0*
_class
loc:@conv1b/bias*
validate_shape(*
_output_shapes
:@
~
training/Adam/mul_23MulAdam/beta_1/readtraining/Adam/Variable_4/read*
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
Š
training/Adam/mul_24Multraining/Adam/sub_14Dtraining/Adam/gradients/conv2a/convolution_grad/Conv2DBackpropFilter*'
_output_shapes
:@*
T0
y
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*'
_output_shapes
:@

training/Adam/mul_25MulAdam/beta_2/readtraining/Adam/Variable_50/read*
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

training/Adam/Square_4SquareDtraining/Adam/gradients/conv2a/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
{
training/Adam/mul_26Multraining/Adam/sub_15training/Adam/Square_4*
T0*'
_output_shapes
:@
y
training/Adam/add_15Addtraining/Adam/mul_25training/Adam/mul_26*
T0*'
_output_shapes
:@
x
training/Adam/mul_27Multraining/Adam/mul_2training/Adam/add_14*
T0*'
_output_shapes
:@
[
training/Adam/Const_10Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_11Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_15training/Adam/Const_11*
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
training/Adam/add_16/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
{
training/Adam/add_16Addtraining/Adam/Sqrt_5training/Adam/add_16/y*'
_output_shapes
:@*
T0

training/Adam/truediv_6RealDivtraining/Adam/mul_27training/Adam/add_16*'
_output_shapes
:@*
T0
z
training/Adam/sub_16Subconv2a/kernel/readtraining/Adam/truediv_6*
T0*'
_output_shapes
:@
Ů
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_14*'
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(
Ű
training/Adam/Assign_13Assigntraining/Adam/Variable_50training/Adam/add_15*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_50*
validate_shape(*'
_output_shapes
:@
Ă
training/Adam/Assign_14Assignconv2a/kerneltraining/Adam/sub_16*
T0* 
_class
loc:@conv2a/kernel*
validate_shape(*'
_output_shapes
:@*
use_locking(
r
training/Adam/mul_28MulAdam/beta_1/readtraining/Adam/Variable_5/read*
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

training/Adam/mul_29Multraining/Adam/sub_177training/Adam/gradients/conv2a/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes	
:
s
training/Adam/mul_30MulAdam/beta_2/readtraining/Adam/Variable_51/read*
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

training/Adam/Square_5Square7training/Adam/gradients/conv2a/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/mul_31Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes	
:
m
training/Adam/add_18Addtraining/Adam/mul_30training/Adam/mul_31*
T0*
_output_shapes	
:
l
training/Adam/mul_32Multraining/Adam/mul_2training/Adam/add_17*
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
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_18training/Adam/Const_13*
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
training/Adam/add_19/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
o
training/Adam/add_19Addtraining/Adam/Sqrt_6training/Adam/add_19/y*
T0*
_output_shapes	
:
t
training/Adam/truediv_7RealDivtraining/Adam/mul_32training/Adam/add_19*
T0*
_output_shapes	
:
l
training/Adam/sub_19Subconv2a/bias/readtraining/Adam/truediv_7*
T0*
_output_shapes	
:
Í
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_17*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5
Ď
training/Adam/Assign_16Assigntraining/Adam/Variable_51training/Adam/add_18*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_51*
validate_shape(*
_output_shapes	
:
ł
training/Adam/Assign_17Assignconv2a/biastraining/Adam/sub_19*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv2a/bias*
validate_shape(

training/Adam/mul_33MulAdam/beta_1/readtraining/Adam/Variable_6/read*(
_output_shapes
:*
T0
[
training/Adam/sub_20/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
_output_shapes
: *
T0
Ş
training/Adam/mul_34Multraining/Adam/sub_20Dtraining/Adam/gradients/conv2b/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
z
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*(
_output_shapes
:*
T0

training/Adam/mul_35MulAdam/beta_2/readtraining/Adam/Variable_52/read*
T0*(
_output_shapes
:
[
training/Adam/sub_21/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_6SquareDtraining/Adam/gradients/conv2b/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
|
training/Adam/mul_36Multraining/Adam/sub_21training/Adam/Square_6*
T0*(
_output_shapes
:
z
training/Adam/add_21Addtraining/Adam/mul_35training/Adam/mul_36*
T0*(
_output_shapes
:
y
training/Adam/mul_37Multraining/Adam/mul_2training/Adam/add_20*(
_output_shapes
:*
T0
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
%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_21training/Adam/Const_15*
T0*(
_output_shapes
:
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
training/Adam/add_22/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
|
training/Adam/add_22Addtraining/Adam/Sqrt_7training/Adam/add_22/y*
T0*(
_output_shapes
:

training/Adam/truediv_8RealDivtraining/Adam/mul_37training/Adam/add_22*(
_output_shapes
:*
T0
{
training/Adam/sub_22Subconv2b/kernel/readtraining/Adam/truediv_8*
T0*(
_output_shapes
:
Ú
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_20*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_19Assigntraining/Adam/Variable_52training/Adam/add_21*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_52*
validate_shape(*(
_output_shapes
:
Ä
training/Adam/Assign_20Assignconv2b/kerneltraining/Adam/sub_22*
T0* 
_class
loc:@conv2b/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
r
training/Adam/mul_38MulAdam/beta_1/readtraining/Adam/Variable_7/read*
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
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_39Multraining/Adam/sub_237training/Adam/gradients/conv2b/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes	
:
s
training/Adam/mul_40MulAdam/beta_2/readtraining/Adam/Variable_53/read*
T0*
_output_shapes	
:
[
training/Adam/sub_24/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_7Square7training/Adam/gradients/conv2b/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/mul_41Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes	
:
m
training/Adam/add_24Addtraining/Adam/mul_40training/Adam/mul_41*
T0*
_output_shapes	
:
l
training/Adam/mul_42Multraining/Adam/mul_2training/Adam/add_23*
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
%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_24training/Adam/Const_17*
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
training/Adam/add_25/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
o
training/Adam/add_25Addtraining/Adam/Sqrt_8training/Adam/add_25/y*
T0*
_output_shapes	
:
t
training/Adam/truediv_9RealDivtraining/Adam/mul_42training/Adam/add_25*
_output_shapes	
:*
T0
l
training/Adam/sub_25Subconv2b/bias/readtraining/Adam/truediv_9*
T0*
_output_shapes	
:
Í
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_23*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7
Ď
training/Adam/Assign_22Assigntraining/Adam/Variable_53training/Adam/add_24*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_53*
validate_shape(*
_output_shapes	
:
ł
training/Adam/Assign_23Assignconv2b/biastraining/Adam/sub_25*
use_locking(*
T0*
_class
loc:@conv2b/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_43MulAdam/beta_1/readtraining/Adam/Variable_8/read*
T0*(
_output_shapes
:
[
training/Adam/sub_26/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_26Subtraining/Adam/sub_26/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ş
training/Adam/mul_44Multraining/Adam/sub_26Dtraining/Adam/gradients/conv3a/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
z
training/Adam/add_26Addtraining/Adam/mul_43training/Adam/mul_44*
T0*(
_output_shapes
:

training/Adam/mul_45MulAdam/beta_2/readtraining/Adam/Variable_54/read*(
_output_shapes
:*
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

training/Adam/Square_8SquareDtraining/Adam/gradients/conv3a/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
|
training/Adam/mul_46Multraining/Adam/sub_27training/Adam/Square_8*
T0*(
_output_shapes
:
z
training/Adam/add_27Addtraining/Adam/mul_45training/Adam/mul_46*
T0*(
_output_shapes
:
y
training/Adam/mul_47Multraining/Adam/mul_2training/Adam/add_26*
T0*(
_output_shapes
:
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

%training/Adam/clip_by_value_9/MinimumMinimumtraining/Adam/add_27training/Adam/Const_19*
T0*(
_output_shapes
:

training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_18*
T0*(
_output_shapes
:
n
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
T0*(
_output_shapes
:
[
training/Adam/add_28/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
|
training/Adam/add_28Addtraining/Adam/Sqrt_9training/Adam/add_28/y*(
_output_shapes
:*
T0

training/Adam/truediv_10RealDivtraining/Adam/mul_47training/Adam/add_28*(
_output_shapes
:*
T0
|
training/Adam/sub_28Subconv3a/kernel/readtraining/Adam/truediv_10*(
_output_shapes
:*
T0
Ú
training/Adam/Assign_24Assigntraining/Adam/Variable_8training/Adam/add_26*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_25Assigntraining/Adam/Variable_54training/Adam/add_27*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_54*
validate_shape(*(
_output_shapes
:
Ä
training/Adam/Assign_26Assignconv3a/kerneltraining/Adam/sub_28*
use_locking(*
T0* 
_class
loc:@conv3a/kernel*
validate_shape(*(
_output_shapes
:
r
training/Adam/mul_48MulAdam/beta_1/readtraining/Adam/Variable_9/read*
T0*
_output_shapes	
:
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

training/Adam/mul_49Multraining/Adam/sub_297training/Adam/gradients/conv3a/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_29Addtraining/Adam/mul_48training/Adam/mul_49*
T0*
_output_shapes	
:
s
training/Adam/mul_50MulAdam/beta_2/readtraining/Adam/Variable_55/read*
T0*
_output_shapes	
:
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

training/Adam/Square_9Square7training/Adam/gradients/conv3a/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/mul_51Multraining/Adam/sub_30training/Adam/Square_9*
_output_shapes	
:*
T0
m
training/Adam/add_30Addtraining/Adam/mul_50training/Adam/mul_51*
_output_shapes	
:*
T0
l
training/Adam/mul_52Multraining/Adam/mul_2training/Adam/add_29*
T0*
_output_shapes	
:
[
training/Adam/Const_20Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_21Const*
dtype0*
_output_shapes
: *
valueB
 *  

&training/Adam/clip_by_value_10/MinimumMinimumtraining/Adam/add_30training/Adam/Const_21*
_output_shapes	
:*
T0

training/Adam/clip_by_value_10Maximum&training/Adam/clip_by_value_10/Minimumtraining/Adam/Const_20*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
_output_shapes	
:*
T0
[
training/Adam/add_31/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_31Addtraining/Adam/Sqrt_10training/Adam/add_31/y*
_output_shapes	
:*
T0
u
training/Adam/truediv_11RealDivtraining/Adam/mul_52training/Adam/add_31*
T0*
_output_shapes	
:
m
training/Adam/sub_31Subconv3a/bias/readtraining/Adam/truediv_11*
T0*
_output_shapes	
:
Í
training/Adam/Assign_27Assigntraining/Adam/Variable_9training/Adam/add_29*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_28Assigntraining/Adam/Variable_55training/Adam/add_30*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_55*
validate_shape(
ł
training/Adam/Assign_29Assignconv3a/biastraining/Adam/sub_31*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv3a/bias*
validate_shape(

training/Adam/mul_53MulAdam/beta_1/readtraining/Adam/Variable_10/read*
T0*(
_output_shapes
:
[
training/Adam/sub_32/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_32Subtraining/Adam/sub_32/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ş
training/Adam/mul_54Multraining/Adam/sub_32Dtraining/Adam/gradients/conv3b/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
z
training/Adam/add_32Addtraining/Adam/mul_53training/Adam/mul_54*
T0*(
_output_shapes
:

training/Adam/mul_55MulAdam/beta_2/readtraining/Adam/Variable_56/read*
T0*(
_output_shapes
:
[
training/Adam/sub_33/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_33Subtraining/Adam/sub_33/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_10SquareDtraining/Adam/gradients/conv3b/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
}
training/Adam/mul_56Multraining/Adam/sub_33training/Adam/Square_10*
T0*(
_output_shapes
:
z
training/Adam/add_33Addtraining/Adam/mul_55training/Adam/mul_56*(
_output_shapes
:*
T0
y
training/Adam/mul_57Multraining/Adam/mul_2training/Adam/add_32*(
_output_shapes
:*
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

&training/Adam/clip_by_value_11/MinimumMinimumtraining/Adam/add_33training/Adam/Const_23*(
_output_shapes
:*
T0

training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_22*
T0*(
_output_shapes
:
p
training/Adam/Sqrt_11Sqrttraining/Adam/clip_by_value_11*
T0*(
_output_shapes
:
[
training/Adam/add_34/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_34Addtraining/Adam/Sqrt_11training/Adam/add_34/y*
T0*(
_output_shapes
:

training/Adam/truediv_12RealDivtraining/Adam/mul_57training/Adam/add_34*
T0*(
_output_shapes
:
|
training/Adam/sub_34Subconv3b/kernel/readtraining/Adam/truediv_12*(
_output_shapes
:*
T0
Ü
training/Adam/Assign_30Assigntraining/Adam/Variable_10training/Adam/add_32*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_31Assigntraining/Adam/Variable_56training/Adam/add_33*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_56*
validate_shape(*(
_output_shapes
:
Ä
training/Adam/Assign_32Assignconv3b/kerneltraining/Adam/sub_34* 
_class
loc:@conv3b/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
s
training/Adam/mul_58MulAdam/beta_1/readtraining/Adam/Variable_11/read*
T0*
_output_shapes	
:
[
training/Adam/sub_35/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_35Subtraining/Adam/sub_35/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_59Multraining/Adam/sub_357training/Adam/gradients/conv3b/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
m
training/Adam/add_35Addtraining/Adam/mul_58training/Adam/mul_59*
_output_shapes	
:*
T0
s
training/Adam/mul_60MulAdam/beta_2/readtraining/Adam/Variable_57/read*
T0*
_output_shapes	
:
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

training/Adam/Square_11Square7training/Adam/gradients/conv3b/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
p
training/Adam/mul_61Multraining/Adam/sub_36training/Adam/Square_11*
T0*
_output_shapes	
:
m
training/Adam/add_36Addtraining/Adam/mul_60training/Adam/mul_61*
_output_shapes	
:*
T0
l
training/Adam/mul_62Multraining/Adam/mul_2training/Adam/add_35*
T0*
_output_shapes	
:
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

&training/Adam/clip_by_value_12/MinimumMinimumtraining/Adam/add_36training/Adam/Const_25*
_output_shapes	
:*
T0

training/Adam/clip_by_value_12Maximum&training/Adam/clip_by_value_12/Minimumtraining/Adam/Const_24*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_12Sqrttraining/Adam/clip_by_value_12*
T0*
_output_shapes	
:
[
training/Adam/add_37/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_37Addtraining/Adam/Sqrt_12training/Adam/add_37/y*
_output_shapes	
:*
T0
u
training/Adam/truediv_13RealDivtraining/Adam/mul_62training/Adam/add_37*
T0*
_output_shapes	
:
m
training/Adam/sub_37Subconv3b/bias/readtraining/Adam/truediv_13*
_output_shapes	
:*
T0
Ď
training/Adam/Assign_33Assigntraining/Adam/Variable_11training/Adam/add_35*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_34Assigntraining/Adam/Variable_57training/Adam/add_36*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_57*
validate_shape(*
_output_shapes	
:
ł
training/Adam/Assign_35Assignconv3b/biastraining/Adam/sub_37*
T0*
_class
loc:@conv3b/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

training/Adam/mul_63MulAdam/beta_1/readtraining/Adam/Variable_12/read*(
_output_shapes
:*
T0
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
training/Adam/mul_64Multraining/Adam/sub_38Dtraining/Adam/gradients/conv4a/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
z
training/Adam/add_38Addtraining/Adam/mul_63training/Adam/mul_64*
T0*(
_output_shapes
:

training/Adam/mul_65MulAdam/beta_2/readtraining/Adam/Variable_58/read*
T0*(
_output_shapes
:
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
training/Adam/Square_12SquareDtraining/Adam/gradients/conv4a/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
}
training/Adam/mul_66Multraining/Adam/sub_39training/Adam/Square_12*
T0*(
_output_shapes
:
z
training/Adam/add_39Addtraining/Adam/mul_65training/Adam/mul_66*
T0*(
_output_shapes
:
y
training/Adam/mul_67Multraining/Adam/mul_2training/Adam/add_38*(
_output_shapes
:*
T0
[
training/Adam/Const_26Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_27Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_13/MinimumMinimumtraining/Adam/add_39training/Adam/Const_27*
T0*(
_output_shapes
:

training/Adam/clip_by_value_13Maximum&training/Adam/clip_by_value_13/Minimumtraining/Adam/Const_26*
T0*(
_output_shapes
:
p
training/Adam/Sqrt_13Sqrttraining/Adam/clip_by_value_13*(
_output_shapes
:*
T0
[
training/Adam/add_40/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_40Addtraining/Adam/Sqrt_13training/Adam/add_40/y*
T0*(
_output_shapes
:

training/Adam/truediv_14RealDivtraining/Adam/mul_67training/Adam/add_40*
T0*(
_output_shapes
:
|
training/Adam/sub_40Subconv4a/kernel/readtraining/Adam/truediv_14*(
_output_shapes
:*
T0
Ü
training/Adam/Assign_36Assigntraining/Adam/Variable_12training/Adam/add_38*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12
Ü
training/Adam/Assign_37Assigntraining/Adam/Variable_58training/Adam/add_39*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_58*
validate_shape(*(
_output_shapes
:
Ä
training/Adam/Assign_38Assignconv4a/kerneltraining/Adam/sub_40*
use_locking(*
T0* 
_class
loc:@conv4a/kernel*
validate_shape(*(
_output_shapes
:
s
training/Adam/mul_68MulAdam/beta_1/readtraining/Adam/Variable_13/read*
_output_shapes	
:*
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

training/Adam/mul_69Multraining/Adam/sub_417training/Adam/gradients/conv4a/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
m
training/Adam/add_41Addtraining/Adam/mul_68training/Adam/mul_69*
T0*
_output_shapes	
:
s
training/Adam/mul_70MulAdam/beta_2/readtraining/Adam/Variable_59/read*
T0*
_output_shapes	
:
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

training/Adam/Square_13Square7training/Adam/gradients/conv4a/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
p
training/Adam/mul_71Multraining/Adam/sub_42training/Adam/Square_13*
_output_shapes	
:*
T0
m
training/Adam/add_42Addtraining/Adam/mul_70training/Adam/mul_71*
_output_shapes	
:*
T0
l
training/Adam/mul_72Multraining/Adam/mul_2training/Adam/add_41*
_output_shapes	
:*
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

&training/Adam/clip_by_value_14/MinimumMinimumtraining/Adam/add_42training/Adam/Const_29*
T0*
_output_shapes	
:

training/Adam/clip_by_value_14Maximum&training/Adam/clip_by_value_14/Minimumtraining/Adam/Const_28*
_output_shapes	
:*
T0
c
training/Adam/Sqrt_14Sqrttraining/Adam/clip_by_value_14*
T0*
_output_shapes	
:
[
training/Adam/add_43/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_43Addtraining/Adam/Sqrt_14training/Adam/add_43/y*
T0*
_output_shapes	
:
u
training/Adam/truediv_15RealDivtraining/Adam/mul_72training/Adam/add_43*
T0*
_output_shapes	
:
m
training/Adam/sub_43Subconv4a/bias/readtraining/Adam/truediv_15*
T0*
_output_shapes	
:
Ď
training/Adam/Assign_39Assigntraining/Adam/Variable_13training/Adam/add_41*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_40Assigntraining/Adam/Variable_59training/Adam/add_42*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_59*
validate_shape(
ł
training/Adam/Assign_41Assignconv4a/biastraining/Adam/sub_43*
use_locking(*
T0*
_class
loc:@conv4a/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_73MulAdam/beta_1/readtraining/Adam/Variable_14/read*(
_output_shapes
:*
T0
[
training/Adam/sub_44/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_44Subtraining/Adam/sub_44/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ş
training/Adam/mul_74Multraining/Adam/sub_44Dtraining/Adam/gradients/conv4b/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
z
training/Adam/add_44Addtraining/Adam/mul_73training/Adam/mul_74*
T0*(
_output_shapes
:

training/Adam/mul_75MulAdam/beta_2/readtraining/Adam/Variable_60/read*
T0*(
_output_shapes
:
[
training/Adam/sub_45/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_45Subtraining/Adam/sub_45/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_14SquareDtraining/Adam/gradients/conv4b/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
}
training/Adam/mul_76Multraining/Adam/sub_45training/Adam/Square_14*(
_output_shapes
:*
T0
z
training/Adam/add_45Addtraining/Adam/mul_75training/Adam/mul_76*(
_output_shapes
:*
T0
y
training/Adam/mul_77Multraining/Adam/mul_2training/Adam/add_44*(
_output_shapes
:*
T0
[
training/Adam/Const_30Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_31Const*
_output_shapes
: *
valueB
 *  *
dtype0

&training/Adam/clip_by_value_15/MinimumMinimumtraining/Adam/add_45training/Adam/Const_31*
T0*(
_output_shapes
:

training/Adam/clip_by_value_15Maximum&training/Adam/clip_by_value_15/Minimumtraining/Adam/Const_30*(
_output_shapes
:*
T0
p
training/Adam/Sqrt_15Sqrttraining/Adam/clip_by_value_15*
T0*(
_output_shapes
:
[
training/Adam/add_46/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_46Addtraining/Adam/Sqrt_15training/Adam/add_46/y*
T0*(
_output_shapes
:

training/Adam/truediv_16RealDivtraining/Adam/mul_77training/Adam/add_46*
T0*(
_output_shapes
:
|
training/Adam/sub_46Subconv4b/kernel/readtraining/Adam/truediv_16*(
_output_shapes
:*
T0
Ü
training/Adam/Assign_42Assigntraining/Adam/Variable_14training/Adam/add_44*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14
Ü
training/Adam/Assign_43Assigntraining/Adam/Variable_60training/Adam/add_45*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_60*
validate_shape(*(
_output_shapes
:
Ä
training/Adam/Assign_44Assignconv4b/kerneltraining/Adam/sub_46*
use_locking(*
T0* 
_class
loc:@conv4b/kernel*
validate_shape(*(
_output_shapes
:
s
training/Adam/mul_78MulAdam/beta_1/readtraining/Adam/Variable_15/read*
_output_shapes	
:*
T0
[
training/Adam/sub_47/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_47Subtraining/Adam/sub_47/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_79Multraining/Adam/sub_477training/Adam/gradients/conv4b/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_47Addtraining/Adam/mul_78training/Adam/mul_79*
_output_shapes	
:*
T0
s
training/Adam/mul_80MulAdam/beta_2/readtraining/Adam/Variable_61/read*
T0*
_output_shapes	
:
[
training/Adam/sub_48/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_48Subtraining/Adam/sub_48/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_15Square7training/Adam/gradients/conv4b/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
p
training/Adam/mul_81Multraining/Adam/sub_48training/Adam/Square_15*
_output_shapes	
:*
T0
m
training/Adam/add_48Addtraining/Adam/mul_80training/Adam/mul_81*
T0*
_output_shapes	
:
l
training/Adam/mul_82Multraining/Adam/mul_2training/Adam/add_47*
T0*
_output_shapes	
:
[
training/Adam/Const_32Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_33Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_16/MinimumMinimumtraining/Adam/add_48training/Adam/Const_33*
T0*
_output_shapes	
:

training/Adam/clip_by_value_16Maximum&training/Adam/clip_by_value_16/Minimumtraining/Adam/Const_32*
_output_shapes	
:*
T0
c
training/Adam/Sqrt_16Sqrttraining/Adam/clip_by_value_16*
T0*
_output_shapes	
:
[
training/Adam/add_49/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_49Addtraining/Adam/Sqrt_16training/Adam/add_49/y*
T0*
_output_shapes	
:
u
training/Adam/truediv_17RealDivtraining/Adam/mul_82training/Adam/add_49*
_output_shapes	
:*
T0
m
training/Adam/sub_49Subconv4b/bias/readtraining/Adam/truediv_17*
T0*
_output_shapes	
:
Ď
training/Adam/Assign_45Assigntraining/Adam/Variable_15training/Adam/add_47*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_46Assigntraining/Adam/Variable_61training/Adam/add_48*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_61*
validate_shape(
ł
training/Adam/Assign_47Assignconv4b/biastraining/Adam/sub_49*
use_locking(*
T0*
_class
loc:@conv4b/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_83MulAdam/beta_1/readtraining/Adam/Variable_16/read*
T0*(
_output_shapes
:
[
training/Adam/sub_50/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_50Subtraining/Adam/sub_50/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ş
training/Adam/mul_84Multraining/Adam/sub_50Dtraining/Adam/gradients/conv5a/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
z
training/Adam/add_50Addtraining/Adam/mul_83training/Adam/mul_84*
T0*(
_output_shapes
:

training/Adam/mul_85MulAdam/beta_2/readtraining/Adam/Variable_62/read*
T0*(
_output_shapes
:
[
training/Adam/sub_51/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_51Subtraining/Adam/sub_51/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_16SquareDtraining/Adam/gradients/conv5a/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
}
training/Adam/mul_86Multraining/Adam/sub_51training/Adam/Square_16*
T0*(
_output_shapes
:
z
training/Adam/add_51Addtraining/Adam/mul_85training/Adam/mul_86*
T0*(
_output_shapes
:
y
training/Adam/mul_87Multraining/Adam/mul_2training/Adam/add_50*
T0*(
_output_shapes
:
[
training/Adam/Const_34Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_35Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_17/MinimumMinimumtraining/Adam/add_51training/Adam/Const_35*
T0*(
_output_shapes
:

training/Adam/clip_by_value_17Maximum&training/Adam/clip_by_value_17/Minimumtraining/Adam/Const_34*(
_output_shapes
:*
T0
p
training/Adam/Sqrt_17Sqrttraining/Adam/clip_by_value_17*
T0*(
_output_shapes
:
[
training/Adam/add_52/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_52Addtraining/Adam/Sqrt_17training/Adam/add_52/y*
T0*(
_output_shapes
:

training/Adam/truediv_18RealDivtraining/Adam/mul_87training/Adam/add_52*(
_output_shapes
:*
T0
|
training/Adam/sub_52Subconv5a/kernel/readtraining/Adam/truediv_18*(
_output_shapes
:*
T0
Ü
training/Adam/Assign_48Assigntraining/Adam/Variable_16training/Adam/add_50*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_49Assigntraining/Adam/Variable_62training/Adam/add_51*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_62*
validate_shape(*(
_output_shapes
:
Ä
training/Adam/Assign_50Assignconv5a/kerneltraining/Adam/sub_52*(
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv5a/kernel*
validate_shape(
s
training/Adam/mul_88MulAdam/beta_1/readtraining/Adam/Variable_17/read*
T0*
_output_shapes	
:
[
training/Adam/sub_53/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_53Subtraining/Adam/sub_53/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_89Multraining/Adam/sub_537training/Adam/gradients/conv5a/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
m
training/Adam/add_53Addtraining/Adam/mul_88training/Adam/mul_89*
T0*
_output_shapes	
:
s
training/Adam/mul_90MulAdam/beta_2/readtraining/Adam/Variable_63/read*
T0*
_output_shapes	
:
[
training/Adam/sub_54/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_54Subtraining/Adam/sub_54/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_17Square7training/Adam/gradients/conv5a/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
p
training/Adam/mul_91Multraining/Adam/sub_54training/Adam/Square_17*
_output_shapes	
:*
T0
m
training/Adam/add_54Addtraining/Adam/mul_90training/Adam/mul_91*
T0*
_output_shapes	
:
l
training/Adam/mul_92Multraining/Adam/mul_2training/Adam/add_53*
T0*
_output_shapes	
:
[
training/Adam/Const_36Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_37Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_18/MinimumMinimumtraining/Adam/add_54training/Adam/Const_37*
T0*
_output_shapes	
:

training/Adam/clip_by_value_18Maximum&training/Adam/clip_by_value_18/Minimumtraining/Adam/Const_36*
_output_shapes	
:*
T0
c
training/Adam/Sqrt_18Sqrttraining/Adam/clip_by_value_18*
T0*
_output_shapes	
:
[
training/Adam/add_55/yConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
p
training/Adam/add_55Addtraining/Adam/Sqrt_18training/Adam/add_55/y*
T0*
_output_shapes	
:
u
training/Adam/truediv_19RealDivtraining/Adam/mul_92training/Adam/add_55*
T0*
_output_shapes	
:
m
training/Adam/sub_55Subconv5a/bias/readtraining/Adam/truediv_19*
_output_shapes	
:*
T0
Ď
training/Adam/Assign_51Assigntraining/Adam/Variable_17training/Adam/add_53*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_52Assigntraining/Adam/Variable_63training/Adam/add_54*,
_class"
 loc:@training/Adam/Variable_63*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ł
training/Adam/Assign_53Assignconv5a/biastraining/Adam/sub_55*
use_locking(*
T0*
_class
loc:@conv5a/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_93MulAdam/beta_1/readtraining/Adam/Variable_18/read*
T0*(
_output_shapes
:
[
training/Adam/sub_56/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_56Subtraining/Adam/sub_56/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ş
training/Adam/mul_94Multraining/Adam/sub_56Dtraining/Adam/gradients/conv5b/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
z
training/Adam/add_56Addtraining/Adam/mul_93training/Adam/mul_94*
T0*(
_output_shapes
:

training/Adam/mul_95MulAdam/beta_2/readtraining/Adam/Variable_64/read*(
_output_shapes
:*
T0
[
training/Adam/sub_57/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_57Subtraining/Adam/sub_57/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_18SquareDtraining/Adam/gradients/conv5b/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
}
training/Adam/mul_96Multraining/Adam/sub_57training/Adam/Square_18*(
_output_shapes
:*
T0
z
training/Adam/add_57Addtraining/Adam/mul_95training/Adam/mul_96*(
_output_shapes
:*
T0
y
training/Adam/mul_97Multraining/Adam/mul_2training/Adam/add_56*
T0*(
_output_shapes
:
[
training/Adam/Const_38Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_39Const*
dtype0*
_output_shapes
: *
valueB
 *  

&training/Adam/clip_by_value_19/MinimumMinimumtraining/Adam/add_57training/Adam/Const_39*
T0*(
_output_shapes
:

training/Adam/clip_by_value_19Maximum&training/Adam/clip_by_value_19/Minimumtraining/Adam/Const_38*(
_output_shapes
:*
T0
p
training/Adam/Sqrt_19Sqrttraining/Adam/clip_by_value_19*
T0*(
_output_shapes
:
[
training/Adam/add_58/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_58Addtraining/Adam/Sqrt_19training/Adam/add_58/y*
T0*(
_output_shapes
:

training/Adam/truediv_20RealDivtraining/Adam/mul_97training/Adam/add_58*
T0*(
_output_shapes
:
|
training/Adam/sub_58Subconv5b/kernel/readtraining/Adam/truediv_20*
T0*(
_output_shapes
:
Ü
training/Adam/Assign_54Assigntraining/Adam/Variable_18training/Adam/add_56*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_55Assigntraining/Adam/Variable_64training/Adam/add_57*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_64
Ä
training/Adam/Assign_56Assignconv5b/kerneltraining/Adam/sub_58*
use_locking(*
T0* 
_class
loc:@conv5b/kernel*
validate_shape(*(
_output_shapes
:
s
training/Adam/mul_98MulAdam/beta_1/readtraining/Adam/Variable_19/read*
T0*
_output_shapes	
:
[
training/Adam/sub_59/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_59Subtraining/Adam/sub_59/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_99Multraining/Adam/sub_597training/Adam/gradients/conv5b/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
m
training/Adam/add_59Addtraining/Adam/mul_98training/Adam/mul_99*
_output_shapes	
:*
T0
t
training/Adam/mul_100MulAdam/beta_2/readtraining/Adam/Variable_65/read*
T0*
_output_shapes	
:
[
training/Adam/sub_60/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_60Subtraining/Adam/sub_60/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_19Square7training/Adam/gradients/conv5b/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
q
training/Adam/mul_101Multraining/Adam/sub_60training/Adam/Square_19*
_output_shapes	
:*
T0
o
training/Adam/add_60Addtraining/Adam/mul_100training/Adam/mul_101*
_output_shapes	
:*
T0
m
training/Adam/mul_102Multraining/Adam/mul_2training/Adam/add_59*
T0*
_output_shapes	
:
[
training/Adam/Const_40Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_41Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_20/MinimumMinimumtraining/Adam/add_60training/Adam/Const_41*
_output_shapes	
:*
T0

training/Adam/clip_by_value_20Maximum&training/Adam/clip_by_value_20/Minimumtraining/Adam/Const_40*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_20Sqrttraining/Adam/clip_by_value_20*
_output_shapes	
:*
T0
[
training/Adam/add_61/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_61Addtraining/Adam/Sqrt_20training/Adam/add_61/y*
T0*
_output_shapes	
:
v
training/Adam/truediv_21RealDivtraining/Adam/mul_102training/Adam/add_61*
T0*
_output_shapes	
:
m
training/Adam/sub_61Subconv5b/bias/readtraining/Adam/truediv_21*
T0*
_output_shapes	
:
Ď
training/Adam/Assign_57Assigntraining/Adam/Variable_19training/Adam/add_59*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_58Assigntraining/Adam/Variable_65training/Adam/add_60*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_65
ł
training/Adam/Assign_59Assignconv5b/biastraining/Adam/sub_61*
use_locking(*
T0*
_class
loc:@conv5b/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_103MulAdam/beta_1/readtraining/Adam/Variable_20/read*
T0*(
_output_shapes
:
[
training/Adam/sub_62/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_62Subtraining/Adam/sub_62/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ş
training/Adam/mul_104Multraining/Adam/sub_62Ctraining/Adam/gradients/conv6/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
|
training/Adam/add_62Addtraining/Adam/mul_103training/Adam/mul_104*(
_output_shapes
:*
T0

training/Adam/mul_105MulAdam/beta_2/readtraining/Adam/Variable_66/read*
T0*(
_output_shapes
:
[
training/Adam/sub_63/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_63Subtraining/Adam/sub_63/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_20SquareCtraining/Adam/gradients/conv6/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
~
training/Adam/mul_106Multraining/Adam/sub_63training/Adam/Square_20*
T0*(
_output_shapes
:
|
training/Adam/add_63Addtraining/Adam/mul_105training/Adam/mul_106*
T0*(
_output_shapes
:
z
training/Adam/mul_107Multraining/Adam/mul_2training/Adam/add_62*
T0*(
_output_shapes
:
[
training/Adam/Const_42Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_43Const*
_output_shapes
: *
valueB
 *  *
dtype0

&training/Adam/clip_by_value_21/MinimumMinimumtraining/Adam/add_63training/Adam/Const_43*(
_output_shapes
:*
T0

training/Adam/clip_by_value_21Maximum&training/Adam/clip_by_value_21/Minimumtraining/Adam/Const_42*(
_output_shapes
:*
T0
p
training/Adam/Sqrt_21Sqrttraining/Adam/clip_by_value_21*
T0*(
_output_shapes
:
[
training/Adam/add_64/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_64Addtraining/Adam/Sqrt_21training/Adam/add_64/y*
T0*(
_output_shapes
:

training/Adam/truediv_22RealDivtraining/Adam/mul_107training/Adam/add_64*
T0*(
_output_shapes
:
{
training/Adam/sub_64Subconv6/kernel/readtraining/Adam/truediv_22*(
_output_shapes
:*
T0
Ü
training/Adam/Assign_60Assigntraining/Adam/Variable_20training/Adam/add_62*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20
Ü
training/Adam/Assign_61Assigntraining/Adam/Variable_66training/Adam/add_63*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_66*
validate_shape(*(
_output_shapes
:
Â
training/Adam/Assign_62Assignconv6/kerneltraining/Adam/sub_64*
use_locking(*
T0*
_class
loc:@conv6/kernel*
validate_shape(*(
_output_shapes
:
t
training/Adam/mul_108MulAdam/beta_1/readtraining/Adam/Variable_21/read*
T0*
_output_shapes	
:
[
training/Adam/sub_65/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_65Subtraining/Adam/sub_65/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_109Multraining/Adam/sub_656training/Adam/gradients/conv6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/add_65Addtraining/Adam/mul_108training/Adam/mul_109*
_output_shapes	
:*
T0
t
training/Adam/mul_110MulAdam/beta_2/readtraining/Adam/Variable_67/read*
T0*
_output_shapes	
:
[
training/Adam/sub_66/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_66Subtraining/Adam/sub_66/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_21Square6training/Adam/gradients/conv6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
q
training/Adam/mul_111Multraining/Adam/sub_66training/Adam/Square_21*
T0*
_output_shapes	
:
o
training/Adam/add_66Addtraining/Adam/mul_110training/Adam/mul_111*
_output_shapes	
:*
T0
m
training/Adam/mul_112Multraining/Adam/mul_2training/Adam/add_65*
_output_shapes	
:*
T0
[
training/Adam/Const_44Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_45Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_22/MinimumMinimumtraining/Adam/add_66training/Adam/Const_45*
T0*
_output_shapes	
:

training/Adam/clip_by_value_22Maximum&training/Adam/clip_by_value_22/Minimumtraining/Adam/Const_44*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_22Sqrttraining/Adam/clip_by_value_22*
_output_shapes	
:*
T0
[
training/Adam/add_67/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_67Addtraining/Adam/Sqrt_22training/Adam/add_67/y*
T0*
_output_shapes	
:
v
training/Adam/truediv_23RealDivtraining/Adam/mul_112training/Adam/add_67*
T0*
_output_shapes	
:
l
training/Adam/sub_67Subconv6/bias/readtraining/Adam/truediv_23*
T0*
_output_shapes	
:
Ď
training/Adam/Assign_63Assigntraining/Adam/Variable_21training/Adam/add_65*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ď
training/Adam/Assign_64Assigntraining/Adam/Variable_67training/Adam/add_66*
T0*,
_class"
 loc:@training/Adam/Variable_67*
validate_shape(*
_output_shapes	
:*
use_locking(
ą
training/Adam/Assign_65Assign
conv6/biastraining/Adam/sub_67*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv6/bias

training/Adam/mul_113MulAdam/beta_1/readtraining/Adam/Variable_22/read*
T0*(
_output_shapes
:
[
training/Adam/sub_68/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_68Subtraining/Adam/sub_68/xAdam/beta_1/read*
_output_shapes
: *
T0
Ť
training/Adam/mul_114Multraining/Adam/sub_68Dtraining/Adam/gradients/conv7a/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
|
training/Adam/add_68Addtraining/Adam/mul_113training/Adam/mul_114*(
_output_shapes
:*
T0

training/Adam/mul_115MulAdam/beta_2/readtraining/Adam/Variable_68/read*(
_output_shapes
:*
T0
[
training/Adam/sub_69/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_69Subtraining/Adam/sub_69/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_22SquareDtraining/Adam/gradients/conv7a/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
~
training/Adam/mul_116Multraining/Adam/sub_69training/Adam/Square_22*(
_output_shapes
:*
T0
|
training/Adam/add_69Addtraining/Adam/mul_115training/Adam/mul_116*
T0*(
_output_shapes
:
z
training/Adam/mul_117Multraining/Adam/mul_2training/Adam/add_68*
T0*(
_output_shapes
:
[
training/Adam/Const_46Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_47Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_23/MinimumMinimumtraining/Adam/add_69training/Adam/Const_47*(
_output_shapes
:*
T0

training/Adam/clip_by_value_23Maximum&training/Adam/clip_by_value_23/Minimumtraining/Adam/Const_46*
T0*(
_output_shapes
:
p
training/Adam/Sqrt_23Sqrttraining/Adam/clip_by_value_23*(
_output_shapes
:*
T0
[
training/Adam/add_70/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_70Addtraining/Adam/Sqrt_23training/Adam/add_70/y*
T0*(
_output_shapes
:

training/Adam/truediv_24RealDivtraining/Adam/mul_117training/Adam/add_70*
T0*(
_output_shapes
:
|
training/Adam/sub_70Subconv7a/kernel/readtraining/Adam/truediv_24*(
_output_shapes
:*
T0
Ü
training/Adam/Assign_66Assigntraining/Adam/Variable_22training/Adam/add_68*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_67Assigntraining/Adam/Variable_68training/Adam/add_69*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_68*
validate_shape(*(
_output_shapes
:
Ä
training/Adam/Assign_68Assignconv7a/kerneltraining/Adam/sub_70*
use_locking(*
T0* 
_class
loc:@conv7a/kernel*
validate_shape(*(
_output_shapes
:
t
training/Adam/mul_118MulAdam/beta_1/readtraining/Adam/Variable_23/read*
T0*
_output_shapes	
:
[
training/Adam/sub_71/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_71Subtraining/Adam/sub_71/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_119Multraining/Adam/sub_717training/Adam/gradients/conv7a/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/add_71Addtraining/Adam/mul_118training/Adam/mul_119*
_output_shapes	
:*
T0
t
training/Adam/mul_120MulAdam/beta_2/readtraining/Adam/Variable_69/read*
T0*
_output_shapes	
:
[
training/Adam/sub_72/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_72Subtraining/Adam/sub_72/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_23Square7training/Adam/gradients/conv7a/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
q
training/Adam/mul_121Multraining/Adam/sub_72training/Adam/Square_23*
T0*
_output_shapes	
:
o
training/Adam/add_72Addtraining/Adam/mul_120training/Adam/mul_121*
T0*
_output_shapes	
:
m
training/Adam/mul_122Multraining/Adam/mul_2training/Adam/add_71*
_output_shapes	
:*
T0
[
training/Adam/Const_48Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_49Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_24/MinimumMinimumtraining/Adam/add_72training/Adam/Const_49*
T0*
_output_shapes	
:

training/Adam/clip_by_value_24Maximum&training/Adam/clip_by_value_24/Minimumtraining/Adam/Const_48*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_24Sqrttraining/Adam/clip_by_value_24*
T0*
_output_shapes	
:
[
training/Adam/add_73/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_73Addtraining/Adam/Sqrt_24training/Adam/add_73/y*
T0*
_output_shapes	
:
v
training/Adam/truediv_25RealDivtraining/Adam/mul_122training/Adam/add_73*
_output_shapes	
:*
T0
m
training/Adam/sub_73Subconv7a/bias/readtraining/Adam/truediv_25*
T0*
_output_shapes	
:
Ď
training/Adam/Assign_69Assigntraining/Adam/Variable_23training/Adam/add_71*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ď
training/Adam/Assign_70Assigntraining/Adam/Variable_69training/Adam/add_72*,
_class"
 loc:@training/Adam/Variable_69*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ł
training/Adam/Assign_71Assignconv7a/biastraining/Adam/sub_73*
use_locking(*
T0*
_class
loc:@conv7a/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_123MulAdam/beta_1/readtraining/Adam/Variable_24/read*
T0*(
_output_shapes
:
[
training/Adam/sub_74/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_74Subtraining/Adam/sub_74/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ť
training/Adam/mul_124Multraining/Adam/sub_74Dtraining/Adam/gradients/conv7b/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
|
training/Adam/add_74Addtraining/Adam/mul_123training/Adam/mul_124*
T0*(
_output_shapes
:

training/Adam/mul_125MulAdam/beta_2/readtraining/Adam/Variable_70/read*
T0*(
_output_shapes
:
[
training/Adam/sub_75/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_75Subtraining/Adam/sub_75/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_24SquareDtraining/Adam/gradients/conv7b/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
~
training/Adam/mul_126Multraining/Adam/sub_75training/Adam/Square_24*
T0*(
_output_shapes
:
|
training/Adam/add_75Addtraining/Adam/mul_125training/Adam/mul_126*
T0*(
_output_shapes
:
z
training/Adam/mul_127Multraining/Adam/mul_2training/Adam/add_74*(
_output_shapes
:*
T0
[
training/Adam/Const_50Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_51Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_25/MinimumMinimumtraining/Adam/add_75training/Adam/Const_51*
T0*(
_output_shapes
:

training/Adam/clip_by_value_25Maximum&training/Adam/clip_by_value_25/Minimumtraining/Adam/Const_50*
T0*(
_output_shapes
:
p
training/Adam/Sqrt_25Sqrttraining/Adam/clip_by_value_25*
T0*(
_output_shapes
:
[
training/Adam/add_76/yConst*
_output_shapes
: *
valueB
 *żÖ3*
dtype0
}
training/Adam/add_76Addtraining/Adam/Sqrt_25training/Adam/add_76/y*
T0*(
_output_shapes
:

training/Adam/truediv_26RealDivtraining/Adam/mul_127training/Adam/add_76*
T0*(
_output_shapes
:
|
training/Adam/sub_76Subconv7b/kernel/readtraining/Adam/truediv_26*
T0*(
_output_shapes
:
Ü
training/Adam/Assign_72Assigntraining/Adam/Variable_24training/Adam/add_74*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24
Ü
training/Adam/Assign_73Assigntraining/Adam/Variable_70training/Adam/add_75*
T0*,
_class"
 loc:@training/Adam/Variable_70*
validate_shape(*(
_output_shapes
:*
use_locking(
Ä
training/Adam/Assign_74Assignconv7b/kerneltraining/Adam/sub_76*
T0* 
_class
loc:@conv7b/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
t
training/Adam/mul_128MulAdam/beta_1/readtraining/Adam/Variable_25/read*
T0*
_output_shapes	
:
[
training/Adam/sub_77/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_77Subtraining/Adam/sub_77/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_129Multraining/Adam/sub_777training/Adam/gradients/conv7b/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/add_77Addtraining/Adam/mul_128training/Adam/mul_129*
_output_shapes	
:*
T0
t
training/Adam/mul_130MulAdam/beta_2/readtraining/Adam/Variable_71/read*
_output_shapes	
:*
T0
[
training/Adam/sub_78/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_78Subtraining/Adam/sub_78/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_25Square7training/Adam/gradients/conv7b/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
q
training/Adam/mul_131Multraining/Adam/sub_78training/Adam/Square_25*
_output_shapes	
:*
T0
o
training/Adam/add_78Addtraining/Adam/mul_130training/Adam/mul_131*
_output_shapes	
:*
T0
m
training/Adam/mul_132Multraining/Adam/mul_2training/Adam/add_77*
T0*
_output_shapes	
:
[
training/Adam/Const_52Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_53Const*
dtype0*
_output_shapes
: *
valueB
 *  

&training/Adam/clip_by_value_26/MinimumMinimumtraining/Adam/add_78training/Adam/Const_53*
_output_shapes	
:*
T0

training/Adam/clip_by_value_26Maximum&training/Adam/clip_by_value_26/Minimumtraining/Adam/Const_52*
_output_shapes	
:*
T0
c
training/Adam/Sqrt_26Sqrttraining/Adam/clip_by_value_26*
_output_shapes	
:*
T0
[
training/Adam/add_79/yConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
p
training/Adam/add_79Addtraining/Adam/Sqrt_26training/Adam/add_79/y*
T0*
_output_shapes	
:
v
training/Adam/truediv_27RealDivtraining/Adam/mul_132training/Adam/add_79*
T0*
_output_shapes	
:
m
training/Adam/sub_79Subconv7b/bias/readtraining/Adam/truediv_27*
T0*
_output_shapes	
:
Ď
training/Adam/Assign_75Assigntraining/Adam/Variable_25training/Adam/add_77*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25
Ď
training/Adam/Assign_76Assigntraining/Adam/Variable_71training/Adam/add_78*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_71*
validate_shape(*
_output_shapes	
:
ł
training/Adam/Assign_77Assignconv7b/biastraining/Adam/sub_79*
_class
loc:@conv7b/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

training/Adam/mul_133MulAdam/beta_1/readtraining/Adam/Variable_26/read*
T0*(
_output_shapes
:
[
training/Adam/sub_80/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_80Subtraining/Adam/sub_80/xAdam/beta_1/read*
_output_shapes
: *
T0
­
training/Adam/mul_134Multraining/Adam/sub_80Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
|
training/Adam/add_80Addtraining/Adam/mul_133training/Adam/mul_134*
T0*(
_output_shapes
:

training/Adam/mul_135MulAdam/beta_2/readtraining/Adam/Variable_72/read*
T0*(
_output_shapes
:
[
training/Adam/sub_81/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_81Subtraining/Adam/sub_81/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_26SquareFtraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
~
training/Adam/mul_136Multraining/Adam/sub_81training/Adam/Square_26*(
_output_shapes
:*
T0
|
training/Adam/add_81Addtraining/Adam/mul_135training/Adam/mul_136*(
_output_shapes
:*
T0
z
training/Adam/mul_137Multraining/Adam/mul_2training/Adam/add_80*
T0*(
_output_shapes
:
[
training/Adam/Const_54Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_55Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_27/MinimumMinimumtraining/Adam/add_81training/Adam/Const_55*(
_output_shapes
:*
T0

training/Adam/clip_by_value_27Maximum&training/Adam/clip_by_value_27/Minimumtraining/Adam/Const_54*
T0*(
_output_shapes
:
p
training/Adam/Sqrt_27Sqrttraining/Adam/clip_by_value_27*(
_output_shapes
:*
T0
[
training/Adam/add_82/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_82Addtraining/Adam/Sqrt_27training/Adam/add_82/y*
T0*(
_output_shapes
:

training/Adam/truediv_28RealDivtraining/Adam/mul_137training/Adam/add_82*
T0*(
_output_shapes
:
~
training/Adam/sub_82Subconv2d_1/kernel/readtraining/Adam/truediv_28*(
_output_shapes
:*
T0
Ü
training/Adam/Assign_78Assigntraining/Adam/Variable_26training/Adam/add_80*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_26*
validate_shape(
Ü
training/Adam/Assign_79Assigntraining/Adam/Variable_72training/Adam/add_81*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_72*
validate_shape(*(
_output_shapes
:
Č
training/Adam/Assign_80Assignconv2d_1/kerneltraining/Adam/sub_82*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
t
training/Adam/mul_138MulAdam/beta_1/readtraining/Adam/Variable_27/read*
T0*
_output_shapes	
:
[
training/Adam/sub_83/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_83Subtraining/Adam/sub_83/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_139Multraining/Adam/sub_839training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/add_83Addtraining/Adam/mul_138training/Adam/mul_139*
T0*
_output_shapes	
:
t
training/Adam/mul_140MulAdam/beta_2/readtraining/Adam/Variable_73/read*
T0*
_output_shapes	
:
[
training/Adam/sub_84/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_84Subtraining/Adam/sub_84/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_27Square9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
q
training/Adam/mul_141Multraining/Adam/sub_84training/Adam/Square_27*
_output_shapes	
:*
T0
o
training/Adam/add_84Addtraining/Adam/mul_140training/Adam/mul_141*
_output_shapes	
:*
T0
m
training/Adam/mul_142Multraining/Adam/mul_2training/Adam/add_83*
T0*
_output_shapes	
:
[
training/Adam/Const_56Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_57Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_28/MinimumMinimumtraining/Adam/add_84training/Adam/Const_57*
T0*
_output_shapes	
:

training/Adam/clip_by_value_28Maximum&training/Adam/clip_by_value_28/Minimumtraining/Adam/Const_56*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_28Sqrttraining/Adam/clip_by_value_28*
T0*
_output_shapes	
:
[
training/Adam/add_85/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_85Addtraining/Adam/Sqrt_28training/Adam/add_85/y*
T0*
_output_shapes	
:
v
training/Adam/truediv_29RealDivtraining/Adam/mul_142training/Adam/add_85*
T0*
_output_shapes	
:
o
training/Adam/sub_85Subconv2d_1/bias/readtraining/Adam/truediv_29*
_output_shapes	
:*
T0
Ď
training/Adam/Assign_81Assigntraining/Adam/Variable_27training/Adam/add_83*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_27*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_82Assigntraining/Adam/Variable_73training/Adam/add_84*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_73*
validate_shape(*
_output_shapes	
:
ˇ
training/Adam/Assign_83Assignconv2d_1/biastraining/Adam/sub_85*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_143MulAdam/beta_1/readtraining/Adam/Variable_28/read*
T0*(
_output_shapes
:
[
training/Adam/sub_86/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_86Subtraining/Adam/sub_86/xAdam/beta_1/read*
T0*
_output_shapes
: 
­
training/Adam/mul_144Multraining/Adam/sub_86Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
|
training/Adam/add_86Addtraining/Adam/mul_143training/Adam/mul_144*(
_output_shapes
:*
T0

training/Adam/mul_145MulAdam/beta_2/readtraining/Adam/Variable_74/read*
T0*(
_output_shapes
:
[
training/Adam/sub_87/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_87Subtraining/Adam/sub_87/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_28SquareFtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
~
training/Adam/mul_146Multraining/Adam/sub_87training/Adam/Square_28*(
_output_shapes
:*
T0
|
training/Adam/add_87Addtraining/Adam/mul_145training/Adam/mul_146*(
_output_shapes
:*
T0
z
training/Adam/mul_147Multraining/Adam/mul_2training/Adam/add_86*
T0*(
_output_shapes
:
[
training/Adam/Const_58Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_59Const*
_output_shapes
: *
valueB
 *  *
dtype0

&training/Adam/clip_by_value_29/MinimumMinimumtraining/Adam/add_87training/Adam/Const_59*
T0*(
_output_shapes
:

training/Adam/clip_by_value_29Maximum&training/Adam/clip_by_value_29/Minimumtraining/Adam/Const_58*(
_output_shapes
:*
T0
p
training/Adam/Sqrt_29Sqrttraining/Adam/clip_by_value_29*(
_output_shapes
:*
T0
[
training/Adam/add_88/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_88Addtraining/Adam/Sqrt_29training/Adam/add_88/y*(
_output_shapes
:*
T0

training/Adam/truediv_30RealDivtraining/Adam/mul_147training/Adam/add_88*
T0*(
_output_shapes
:
~
training/Adam/sub_88Subconv2d_2/kernel/readtraining/Adam/truediv_30*
T0*(
_output_shapes
:
Ü
training/Adam/Assign_84Assigntraining/Adam/Variable_28training/Adam/add_86*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_28
Ü
training/Adam/Assign_85Assigntraining/Adam/Variable_74training/Adam/add_87*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_74*
validate_shape(*(
_output_shapes
:
Č
training/Adam/Assign_86Assignconv2d_2/kerneltraining/Adam/sub_88*(
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(
t
training/Adam/mul_148MulAdam/beta_1/readtraining/Adam/Variable_29/read*
T0*
_output_shapes	
:
[
training/Adam/sub_89/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_89Subtraining/Adam/sub_89/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_149Multraining/Adam/sub_899training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/add_89Addtraining/Adam/mul_148training/Adam/mul_149*
T0*
_output_shapes	
:
t
training/Adam/mul_150MulAdam/beta_2/readtraining/Adam/Variable_75/read*
T0*
_output_shapes	
:
[
training/Adam/sub_90/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_90Subtraining/Adam/sub_90/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_29Square9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
q
training/Adam/mul_151Multraining/Adam/sub_90training/Adam/Square_29*
T0*
_output_shapes	
:
o
training/Adam/add_90Addtraining/Adam/mul_150training/Adam/mul_151*
T0*
_output_shapes	
:
m
training/Adam/mul_152Multraining/Adam/mul_2training/Adam/add_89*
T0*
_output_shapes	
:
[
training/Adam/Const_60Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_61Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_30/MinimumMinimumtraining/Adam/add_90training/Adam/Const_61*
_output_shapes	
:*
T0

training/Adam/clip_by_value_30Maximum&training/Adam/clip_by_value_30/Minimumtraining/Adam/Const_60*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_30Sqrttraining/Adam/clip_by_value_30*
_output_shapes	
:*
T0
[
training/Adam/add_91/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_91Addtraining/Adam/Sqrt_30training/Adam/add_91/y*
_output_shapes	
:*
T0
v
training/Adam/truediv_31RealDivtraining/Adam/mul_152training/Adam/add_91*
_output_shapes	
:*
T0
o
training/Adam/sub_91Subconv2d_2/bias/readtraining/Adam/truediv_31*
_output_shapes	
:*
T0
Ď
training/Adam/Assign_87Assigntraining/Adam/Variable_29training/Adam/add_89*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_29*
validate_shape(
Ď
training/Adam/Assign_88Assigntraining/Adam/Variable_75training/Adam/add_90*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_75*
validate_shape(
ˇ
training/Adam/Assign_89Assignconv2d_2/biastraining/Adam/sub_91*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_153MulAdam/beta_1/readtraining/Adam/Variable_30/read*
T0*(
_output_shapes
:
[
training/Adam/sub_92/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_92Subtraining/Adam/sub_92/xAdam/beta_1/read*
T0*
_output_shapes
: 
­
training/Adam/mul_154Multraining/Adam/sub_92Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
|
training/Adam/add_92Addtraining/Adam/mul_153training/Adam/mul_154*
T0*(
_output_shapes
:

training/Adam/mul_155MulAdam/beta_2/readtraining/Adam/Variable_76/read*(
_output_shapes
:*
T0
[
training/Adam/sub_93/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_93Subtraining/Adam/sub_93/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_30SquareFtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
~
training/Adam/mul_156Multraining/Adam/sub_93training/Adam/Square_30*(
_output_shapes
:*
T0
|
training/Adam/add_93Addtraining/Adam/mul_155training/Adam/mul_156*
T0*(
_output_shapes
:
z
training/Adam/mul_157Multraining/Adam/mul_2training/Adam/add_92*
T0*(
_output_shapes
:
[
training/Adam/Const_62Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_63Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_31/MinimumMinimumtraining/Adam/add_93training/Adam/Const_63*
T0*(
_output_shapes
:

training/Adam/clip_by_value_31Maximum&training/Adam/clip_by_value_31/Minimumtraining/Adam/Const_62*
T0*(
_output_shapes
:
p
training/Adam/Sqrt_31Sqrttraining/Adam/clip_by_value_31*(
_output_shapes
:*
T0
[
training/Adam/add_94/yConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
}
training/Adam/add_94Addtraining/Adam/Sqrt_31training/Adam/add_94/y*
T0*(
_output_shapes
:

training/Adam/truediv_32RealDivtraining/Adam/mul_157training/Adam/add_94*(
_output_shapes
:*
T0
~
training/Adam/sub_94Subconv2d_3/kernel/readtraining/Adam/truediv_32*
T0*(
_output_shapes
:
Ü
training/Adam/Assign_90Assigntraining/Adam/Variable_30training/Adam/add_92*
T0*,
_class"
 loc:@training/Adam/Variable_30*
validate_shape(*(
_output_shapes
:*
use_locking(
Ü
training/Adam/Assign_91Assigntraining/Adam/Variable_76training/Adam/add_93*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_76*
validate_shape(*(
_output_shapes
:
Č
training/Adam/Assign_92Assignconv2d_3/kerneltraining/Adam/sub_94*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(*(
_output_shapes
:
t
training/Adam/mul_158MulAdam/beta_1/readtraining/Adam/Variable_31/read*
T0*
_output_shapes	
:
[
training/Adam/sub_95/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_95Subtraining/Adam/sub_95/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_159Multraining/Adam/sub_959training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/add_95Addtraining/Adam/mul_158training/Adam/mul_159*
_output_shapes	
:*
T0
t
training/Adam/mul_160MulAdam/beta_2/readtraining/Adam/Variable_77/read*
T0*
_output_shapes	
:
[
training/Adam/sub_96/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_96Subtraining/Adam/sub_96/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_31Square9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
q
training/Adam/mul_161Multraining/Adam/sub_96training/Adam/Square_31*
_output_shapes	
:*
T0
o
training/Adam/add_96Addtraining/Adam/mul_160training/Adam/mul_161*
T0*
_output_shapes	
:
m
training/Adam/mul_162Multraining/Adam/mul_2training/Adam/add_95*
T0*
_output_shapes	
:
[
training/Adam/Const_64Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_65Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_32/MinimumMinimumtraining/Adam/add_96training/Adam/Const_65*
T0*
_output_shapes	
:

training/Adam/clip_by_value_32Maximum&training/Adam/clip_by_value_32/Minimumtraining/Adam/Const_64*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_32Sqrttraining/Adam/clip_by_value_32*
T0*
_output_shapes	
:
[
training/Adam/add_97/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_97Addtraining/Adam/Sqrt_32training/Adam/add_97/y*
_output_shapes	
:*
T0
v
training/Adam/truediv_33RealDivtraining/Adam/mul_162training/Adam/add_97*
T0*
_output_shapes	
:
o
training/Adam/sub_97Subconv2d_3/bias/readtraining/Adam/truediv_33*
_output_shapes	
:*
T0
Ď
training/Adam/Assign_93Assigntraining/Adam/Variable_31training/Adam/add_95*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_31
Ď
training/Adam/Assign_94Assigntraining/Adam/Variable_77training/Adam/add_96*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_77*
validate_shape(*
_output_shapes	
:
ˇ
training/Adam/Assign_95Assignconv2d_3/biastraining/Adam/sub_97*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_163MulAdam/beta_1/readtraining/Adam/Variable_32/read*
T0*(
_output_shapes
:
[
training/Adam/sub_98/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_98Subtraining/Adam/sub_98/xAdam/beta_1/read*
T0*
_output_shapes
: 
­
training/Adam/mul_164Multraining/Adam/sub_98Ftraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
|
training/Adam/add_98Addtraining/Adam/mul_163training/Adam/mul_164*
T0*(
_output_shapes
:

training/Adam/mul_165MulAdam/beta_2/readtraining/Adam/Variable_78/read*
T0*(
_output_shapes
:
[
training/Adam/sub_99/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_99Subtraining/Adam/sub_99/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_32SquareFtraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
~
training/Adam/mul_166Multraining/Adam/sub_99training/Adam/Square_32*(
_output_shapes
:*
T0
|
training/Adam/add_99Addtraining/Adam/mul_165training/Adam/mul_166*(
_output_shapes
:*
T0
z
training/Adam/mul_167Multraining/Adam/mul_2training/Adam/add_98*
T0*(
_output_shapes
:
[
training/Adam/Const_66Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_67Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_33/MinimumMinimumtraining/Adam/add_99training/Adam/Const_67*(
_output_shapes
:*
T0

training/Adam/clip_by_value_33Maximum&training/Adam/clip_by_value_33/Minimumtraining/Adam/Const_66*
T0*(
_output_shapes
:
p
training/Adam/Sqrt_33Sqrttraining/Adam/clip_by_value_33*
T0*(
_output_shapes
:
\
training/Adam/add_100/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 

training/Adam/add_100Addtraining/Adam/Sqrt_33training/Adam/add_100/y*
T0*(
_output_shapes
:

training/Adam/truediv_34RealDivtraining/Adam/mul_167training/Adam/add_100*(
_output_shapes
:*
T0

training/Adam/sub_100Subconv2d_4/kernel/readtraining/Adam/truediv_34*(
_output_shapes
:*
T0
Ü
training/Adam/Assign_96Assigntraining/Adam/Variable_32training/Adam/add_98*,
_class"
 loc:@training/Adam/Variable_32*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Ü
training/Adam/Assign_97Assigntraining/Adam/Variable_78training/Adam/add_99*
T0*,
_class"
 loc:@training/Adam/Variable_78*
validate_shape(*(
_output_shapes
:*
use_locking(
É
training/Adam/Assign_98Assignconv2d_4/kerneltraining/Adam/sub_100*(
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(
t
training/Adam/mul_168MulAdam/beta_1/readtraining/Adam/Variable_33/read*
_output_shapes	
:*
T0
\
training/Adam/sub_101/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
h
training/Adam/sub_101Subtraining/Adam/sub_101/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_169Multraining/Adam/sub_1019training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
p
training/Adam/add_101Addtraining/Adam/mul_168training/Adam/mul_169*
T0*
_output_shapes	
:
t
training/Adam/mul_170MulAdam/beta_2/readtraining/Adam/Variable_79/read*
_output_shapes	
:*
T0
\
training/Adam/sub_102/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_102Subtraining/Adam/sub_102/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_33Square9training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
r
training/Adam/mul_171Multraining/Adam/sub_102training/Adam/Square_33*
T0*
_output_shapes	
:
p
training/Adam/add_102Addtraining/Adam/mul_170training/Adam/mul_171*
T0*
_output_shapes	
:
n
training/Adam/mul_172Multraining/Adam/mul_2training/Adam/add_101*
T0*
_output_shapes	
:
[
training/Adam/Const_68Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_69Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_34/MinimumMinimumtraining/Adam/add_102training/Adam/Const_69*
_output_shapes	
:*
T0

training/Adam/clip_by_value_34Maximum&training/Adam/clip_by_value_34/Minimumtraining/Adam/Const_68*
_output_shapes	
:*
T0
c
training/Adam/Sqrt_34Sqrttraining/Adam/clip_by_value_34*
T0*
_output_shapes	
:
\
training/Adam/add_103/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
r
training/Adam/add_103Addtraining/Adam/Sqrt_34training/Adam/add_103/y*
T0*
_output_shapes	
:
w
training/Adam/truediv_35RealDivtraining/Adam/mul_172training/Adam/add_103*
T0*
_output_shapes	
:
p
training/Adam/sub_103Subconv2d_4/bias/readtraining/Adam/truediv_35*
T0*
_output_shapes	
:
Đ
training/Adam/Assign_99Assigntraining/Adam/Variable_33training/Adam/add_101*
T0*,
_class"
 loc:@training/Adam/Variable_33*
validate_shape(*
_output_shapes	
:*
use_locking(
Ń
training/Adam/Assign_100Assigntraining/Adam/Variable_79training/Adam/add_102*,
_class"
 loc:@training/Adam/Variable_79*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
š
training/Adam/Assign_101Assignconv2d_4/biastraining/Adam/sub_103*
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

training/Adam/mul_173MulAdam/beta_1/readtraining/Adam/Variable_34/read*
T0*(
_output_shapes
:
\
training/Adam/sub_104/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_104Subtraining/Adam/sub_104/xAdam/beta_1/read*
_output_shapes
: *
T0
Ž
training/Adam/mul_174Multraining/Adam/sub_104Ftraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
}
training/Adam/add_104Addtraining/Adam/mul_173training/Adam/mul_174*
T0*(
_output_shapes
:

training/Adam/mul_175MulAdam/beta_2/readtraining/Adam/Variable_80/read*
T0*(
_output_shapes
:
\
training/Adam/sub_105/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_105Subtraining/Adam/sub_105/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_34SquareFtraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0

training/Adam/mul_176Multraining/Adam/sub_105training/Adam/Square_34*
T0*(
_output_shapes
:
}
training/Adam/add_105Addtraining/Adam/mul_175training/Adam/mul_176*
T0*(
_output_shapes
:
{
training/Adam/mul_177Multraining/Adam/mul_2training/Adam/add_104*
T0*(
_output_shapes
:
[
training/Adam/Const_70Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_71Const*
dtype0*
_output_shapes
: *
valueB
 *  

&training/Adam/clip_by_value_35/MinimumMinimumtraining/Adam/add_105training/Adam/Const_71*
T0*(
_output_shapes
:

training/Adam/clip_by_value_35Maximum&training/Adam/clip_by_value_35/Minimumtraining/Adam/Const_70*
T0*(
_output_shapes
:
p
training/Adam/Sqrt_35Sqrttraining/Adam/clip_by_value_35*
T0*(
_output_shapes
:
\
training/Adam/add_106/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 

training/Adam/add_106Addtraining/Adam/Sqrt_35training/Adam/add_106/y*
T0*(
_output_shapes
:

training/Adam/truediv_36RealDivtraining/Adam/mul_177training/Adam/add_106*
T0*(
_output_shapes
:

training/Adam/sub_106Subconv2d_5/kernel/readtraining/Adam/truediv_36*(
_output_shapes
:*
T0
Ţ
training/Adam/Assign_102Assigntraining/Adam/Variable_34training/Adam/add_104*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_34*
validate_shape(*(
_output_shapes
:
Ţ
training/Adam/Assign_103Assigntraining/Adam/Variable_80training/Adam/add_105*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_80*
validate_shape(*(
_output_shapes
:
Ę
training/Adam/Assign_104Assignconv2d_5/kerneltraining/Adam/sub_106*
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel*
validate_shape(*(
_output_shapes
:
t
training/Adam/mul_178MulAdam/beta_1/readtraining/Adam/Variable_35/read*
_output_shapes	
:*
T0
\
training/Adam/sub_107/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_107Subtraining/Adam/sub_107/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_179Multraining/Adam/sub_1079training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
p
training/Adam/add_107Addtraining/Adam/mul_178training/Adam/mul_179*
T0*
_output_shapes	
:
t
training/Adam/mul_180MulAdam/beta_2/readtraining/Adam/Variable_81/read*
_output_shapes	
:*
T0
\
training/Adam/sub_108/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_108Subtraining/Adam/sub_108/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_35Square9training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
r
training/Adam/mul_181Multraining/Adam/sub_108training/Adam/Square_35*
T0*
_output_shapes	
:
p
training/Adam/add_108Addtraining/Adam/mul_180training/Adam/mul_181*
_output_shapes	
:*
T0
n
training/Adam/mul_182Multraining/Adam/mul_2training/Adam/add_107*
_output_shapes	
:*
T0
[
training/Adam/Const_72Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_73Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_36/MinimumMinimumtraining/Adam/add_108training/Adam/Const_73*
_output_shapes	
:*
T0

training/Adam/clip_by_value_36Maximum&training/Adam/clip_by_value_36/Minimumtraining/Adam/Const_72*
_output_shapes	
:*
T0
c
training/Adam/Sqrt_36Sqrttraining/Adam/clip_by_value_36*
T0*
_output_shapes	
:
\
training/Adam/add_109/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
r
training/Adam/add_109Addtraining/Adam/Sqrt_36training/Adam/add_109/y*
_output_shapes	
:*
T0
w
training/Adam/truediv_37RealDivtraining/Adam/mul_182training/Adam/add_109*
T0*
_output_shapes	
:
p
training/Adam/sub_109Subconv2d_5/bias/readtraining/Adam/truediv_37*
T0*
_output_shapes	
:
Ń
training/Adam/Assign_105Assigntraining/Adam/Variable_35training/Adam/add_107*,
_class"
 loc:@training/Adam/Variable_35*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ń
training/Adam/Assign_106Assigntraining/Adam/Variable_81training/Adam/add_108*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_81*
validate_shape(*
_output_shapes	
:
š
training/Adam/Assign_107Assignconv2d_5/biastraining/Adam/sub_109*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias

training/Adam/mul_183MulAdam/beta_1/readtraining/Adam/Variable_36/read*
T0*(
_output_shapes
:
\
training/Adam/sub_110/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_110Subtraining/Adam/sub_110/xAdam/beta_1/read*
_output_shapes
: *
T0
Ž
training/Adam/mul_184Multraining/Adam/sub_110Ftraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
}
training/Adam/add_110Addtraining/Adam/mul_183training/Adam/mul_184*(
_output_shapes
:*
T0

training/Adam/mul_185MulAdam/beta_2/readtraining/Adam/Variable_82/read*
T0*(
_output_shapes
:
\
training/Adam/sub_111/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_111Subtraining/Adam/sub_111/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_36SquareFtraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0

training/Adam/mul_186Multraining/Adam/sub_111training/Adam/Square_36*
T0*(
_output_shapes
:
}
training/Adam/add_111Addtraining/Adam/mul_185training/Adam/mul_186*(
_output_shapes
:*
T0
{
training/Adam/mul_187Multraining/Adam/mul_2training/Adam/add_110*
T0*(
_output_shapes
:
[
training/Adam/Const_74Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_75Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_37/MinimumMinimumtraining/Adam/add_111training/Adam/Const_75*
T0*(
_output_shapes
:

training/Adam/clip_by_value_37Maximum&training/Adam/clip_by_value_37/Minimumtraining/Adam/Const_74*
T0*(
_output_shapes
:
p
training/Adam/Sqrt_37Sqrttraining/Adam/clip_by_value_37*
T0*(
_output_shapes
:
\
training/Adam/add_112/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 

training/Adam/add_112Addtraining/Adam/Sqrt_37training/Adam/add_112/y*
T0*(
_output_shapes
:

training/Adam/truediv_38RealDivtraining/Adam/mul_187training/Adam/add_112*(
_output_shapes
:*
T0

training/Adam/sub_112Subconv2d_6/kernel/readtraining/Adam/truediv_38*
T0*(
_output_shapes
:
Ţ
training/Adam/Assign_108Assigntraining/Adam/Variable_36training/Adam/add_110*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_36*
validate_shape(
Ţ
training/Adam/Assign_109Assigntraining/Adam/Variable_82training/Adam/add_111*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_82*
validate_shape(*(
_output_shapes
:
Ę
training/Adam/Assign_110Assignconv2d_6/kerneltraining/Adam/sub_112*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*(
_output_shapes
:
t
training/Adam/mul_188MulAdam/beta_1/readtraining/Adam/Variable_37/read*
T0*
_output_shapes	
:
\
training/Adam/sub_113/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_113Subtraining/Adam/sub_113/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_189Multraining/Adam/sub_1139training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
p
training/Adam/add_113Addtraining/Adam/mul_188training/Adam/mul_189*
T0*
_output_shapes	
:
t
training/Adam/mul_190MulAdam/beta_2/readtraining/Adam/Variable_83/read*
_output_shapes	
:*
T0
\
training/Adam/sub_114/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_114Subtraining/Adam/sub_114/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_37Square9training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
r
training/Adam/mul_191Multraining/Adam/sub_114training/Adam/Square_37*
_output_shapes	
:*
T0
p
training/Adam/add_114Addtraining/Adam/mul_190training/Adam/mul_191*
T0*
_output_shapes	
:
n
training/Adam/mul_192Multraining/Adam/mul_2training/Adam/add_113*
T0*
_output_shapes	
:
[
training/Adam/Const_76Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_77Const*
_output_shapes
: *
valueB
 *  *
dtype0

&training/Adam/clip_by_value_38/MinimumMinimumtraining/Adam/add_114training/Adam/Const_77*
T0*
_output_shapes	
:

training/Adam/clip_by_value_38Maximum&training/Adam/clip_by_value_38/Minimumtraining/Adam/Const_76*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_38Sqrttraining/Adam/clip_by_value_38*
T0*
_output_shapes	
:
\
training/Adam/add_115/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
r
training/Adam/add_115Addtraining/Adam/Sqrt_38training/Adam/add_115/y*
_output_shapes	
:*
T0
w
training/Adam/truediv_39RealDivtraining/Adam/mul_192training/Adam/add_115*
T0*
_output_shapes	
:
p
training/Adam/sub_115Subconv2d_6/bias/readtraining/Adam/truediv_39*
T0*
_output_shapes	
:
Ń
training/Adam/Assign_111Assigntraining/Adam/Variable_37training/Adam/add_113*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_37*
validate_shape(*
_output_shapes	
:
Ń
training/Adam/Assign_112Assigntraining/Adam/Variable_83training/Adam/add_114*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_83*
validate_shape(
š
training/Adam/Assign_113Assignconv2d_6/biastraining/Adam/sub_115* 
_class
loc:@conv2d_6/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

training/Adam/mul_193MulAdam/beta_1/readtraining/Adam/Variable_38/read*'
_output_shapes
:@*
T0
\
training/Adam/sub_116/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_116Subtraining/Adam/sub_116/xAdam/beta_1/read*
T0*
_output_shapes
: 
­
training/Adam/mul_194Multraining/Adam/sub_116Ftraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropFilter*'
_output_shapes
:@*
T0
|
training/Adam/add_116Addtraining/Adam/mul_193training/Adam/mul_194*'
_output_shapes
:@*
T0

training/Adam/mul_195MulAdam/beta_2/readtraining/Adam/Variable_84/read*'
_output_shapes
:@*
T0
\
training/Adam/sub_117/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_117Subtraining/Adam/sub_117/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_38SquareFtraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
~
training/Adam/mul_196Multraining/Adam/sub_117training/Adam/Square_38*'
_output_shapes
:@*
T0
|
training/Adam/add_117Addtraining/Adam/mul_195training/Adam/mul_196*
T0*'
_output_shapes
:@
z
training/Adam/mul_197Multraining/Adam/mul_2training/Adam/add_116*
T0*'
_output_shapes
:@
[
training/Adam/Const_78Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_79Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_39/MinimumMinimumtraining/Adam/add_117training/Adam/Const_79*'
_output_shapes
:@*
T0

training/Adam/clip_by_value_39Maximum&training/Adam/clip_by_value_39/Minimumtraining/Adam/Const_78*'
_output_shapes
:@*
T0
o
training/Adam/Sqrt_39Sqrttraining/Adam/clip_by_value_39*
T0*'
_output_shapes
:@
\
training/Adam/add_118/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
~
training/Adam/add_118Addtraining/Adam/Sqrt_39training/Adam/add_118/y*
T0*'
_output_shapes
:@

training/Adam/truediv_40RealDivtraining/Adam/mul_197training/Adam/add_118*
T0*'
_output_shapes
:@
~
training/Adam/sub_118Subconv2d_7/kernel/readtraining/Adam/truediv_40*
T0*'
_output_shapes
:@
Ý
training/Adam/Assign_114Assigntraining/Adam/Variable_38training/Adam/add_116*'
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_38*
validate_shape(
Ý
training/Adam/Assign_115Assigntraining/Adam/Variable_84training/Adam/add_117*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_84*
validate_shape(*'
_output_shapes
:@
É
training/Adam/Assign_116Assignconv2d_7/kerneltraining/Adam/sub_118*
use_locking(*
T0*"
_class
loc:@conv2d_7/kernel*
validate_shape(*'
_output_shapes
:@
s
training/Adam/mul_198MulAdam/beta_1/readtraining/Adam/Variable_39/read*
T0*
_output_shapes
:@
\
training/Adam/sub_119/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_119Subtraining/Adam/sub_119/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_199Multraining/Adam/sub_1199training/Adam/gradients/conv2d_7/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
o
training/Adam/add_119Addtraining/Adam/mul_198training/Adam/mul_199*
T0*
_output_shapes
:@
s
training/Adam/mul_200MulAdam/beta_2/readtraining/Adam/Variable_85/read*
T0*
_output_shapes
:@
\
training/Adam/sub_120/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_120Subtraining/Adam/sub_120/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_39Square9training/Adam/gradients/conv2d_7/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
q
training/Adam/mul_201Multraining/Adam/sub_120training/Adam/Square_39*
T0*
_output_shapes
:@
o
training/Adam/add_120Addtraining/Adam/mul_200training/Adam/mul_201*
T0*
_output_shapes
:@
m
training/Adam/mul_202Multraining/Adam/mul_2training/Adam/add_119*
T0*
_output_shapes
:@
[
training/Adam/Const_80Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_81Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_40/MinimumMinimumtraining/Adam/add_120training/Adam/Const_81*
T0*
_output_shapes
:@

training/Adam/clip_by_value_40Maximum&training/Adam/clip_by_value_40/Minimumtraining/Adam/Const_80*
_output_shapes
:@*
T0
b
training/Adam/Sqrt_40Sqrttraining/Adam/clip_by_value_40*
T0*
_output_shapes
:@
\
training/Adam/add_121/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
q
training/Adam/add_121Addtraining/Adam/Sqrt_40training/Adam/add_121/y*
_output_shapes
:@*
T0
v
training/Adam/truediv_41RealDivtraining/Adam/mul_202training/Adam/add_121*
_output_shapes
:@*
T0
o
training/Adam/sub_121Subconv2d_7/bias/readtraining/Adam/truediv_41*
T0*
_output_shapes
:@
Đ
training/Adam/Assign_117Assigntraining/Adam/Variable_39training/Adam/add_119*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_39*
validate_shape(
Đ
training/Adam/Assign_118Assigntraining/Adam/Variable_85training/Adam/add_120*
T0*,
_class"
 loc:@training/Adam/Variable_85*
validate_shape(*
_output_shapes
:@*
use_locking(
¸
training/Adam/Assign_119Assignconv2d_7/biastraining/Adam/sub_121*
use_locking(*
T0* 
_class
loc:@conv2d_7/bias*
validate_shape(*
_output_shapes
:@

training/Adam/mul_203MulAdam/beta_1/readtraining/Adam/Variable_40/read*'
_output_shapes
:@*
T0
\
training/Adam/sub_122/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_122Subtraining/Adam/sub_122/xAdam/beta_1/read*
_output_shapes
: *
T0
­
training/Adam/mul_204Multraining/Adam/sub_122Ftraining/Adam/gradients/conv2d_8/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
|
training/Adam/add_122Addtraining/Adam/mul_203training/Adam/mul_204*
T0*'
_output_shapes
:@

training/Adam/mul_205MulAdam/beta_2/readtraining/Adam/Variable_86/read*'
_output_shapes
:@*
T0
\
training/Adam/sub_123/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_123Subtraining/Adam/sub_123/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_40SquareFtraining/Adam/gradients/conv2d_8/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
~
training/Adam/mul_206Multraining/Adam/sub_123training/Adam/Square_40*
T0*'
_output_shapes
:@
|
training/Adam/add_123Addtraining/Adam/mul_205training/Adam/mul_206*
T0*'
_output_shapes
:@
z
training/Adam/mul_207Multraining/Adam/mul_2training/Adam/add_122*'
_output_shapes
:@*
T0
[
training/Adam/Const_82Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_83Const*
_output_shapes
: *
valueB
 *  *
dtype0

&training/Adam/clip_by_value_41/MinimumMinimumtraining/Adam/add_123training/Adam/Const_83*
T0*'
_output_shapes
:@

training/Adam/clip_by_value_41Maximum&training/Adam/clip_by_value_41/Minimumtraining/Adam/Const_82*
T0*'
_output_shapes
:@
o
training/Adam/Sqrt_41Sqrttraining/Adam/clip_by_value_41*'
_output_shapes
:@*
T0
\
training/Adam/add_124/yConst*
_output_shapes
: *
valueB
 *żÖ3*
dtype0
~
training/Adam/add_124Addtraining/Adam/Sqrt_41training/Adam/add_124/y*
T0*'
_output_shapes
:@

training/Adam/truediv_42RealDivtraining/Adam/mul_207training/Adam/add_124*
T0*'
_output_shapes
:@
~
training/Adam/sub_124Subconv2d_8/kernel/readtraining/Adam/truediv_42*
T0*'
_output_shapes
:@
Ý
training/Adam/Assign_120Assigntraining/Adam/Variable_40training/Adam/add_122*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_40*
validate_shape(*'
_output_shapes
:@
Ý
training/Adam/Assign_121Assigntraining/Adam/Variable_86training/Adam/add_123*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_86*
validate_shape(*'
_output_shapes
:@
É
training/Adam/Assign_122Assignconv2d_8/kerneltraining/Adam/sub_124*
use_locking(*
T0*"
_class
loc:@conv2d_8/kernel*
validate_shape(*'
_output_shapes
:@
s
training/Adam/mul_208MulAdam/beta_1/readtraining/Adam/Variable_41/read*
T0*
_output_shapes
:@
\
training/Adam/sub_125/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
h
training/Adam/sub_125Subtraining/Adam/sub_125/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_209Multraining/Adam/sub_1259training/Adam/gradients/conv2d_8/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
o
training/Adam/add_125Addtraining/Adam/mul_208training/Adam/mul_209*
T0*
_output_shapes
:@
s
training/Adam/mul_210MulAdam/beta_2/readtraining/Adam/Variable_87/read*
_output_shapes
:@*
T0
\
training/Adam/sub_126/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_126Subtraining/Adam/sub_126/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_41Square9training/Adam/gradients/conv2d_8/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
q
training/Adam/mul_211Multraining/Adam/sub_126training/Adam/Square_41*
_output_shapes
:@*
T0
o
training/Adam/add_126Addtraining/Adam/mul_210training/Adam/mul_211*
T0*
_output_shapes
:@
m
training/Adam/mul_212Multraining/Adam/mul_2training/Adam/add_125*
T0*
_output_shapes
:@
[
training/Adam/Const_84Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_85Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_42/MinimumMinimumtraining/Adam/add_126training/Adam/Const_85*
T0*
_output_shapes
:@

training/Adam/clip_by_value_42Maximum&training/Adam/clip_by_value_42/Minimumtraining/Adam/Const_84*
T0*
_output_shapes
:@
b
training/Adam/Sqrt_42Sqrttraining/Adam/clip_by_value_42*
T0*
_output_shapes
:@
\
training/Adam/add_127/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
q
training/Adam/add_127Addtraining/Adam/Sqrt_42training/Adam/add_127/y*
_output_shapes
:@*
T0
v
training/Adam/truediv_43RealDivtraining/Adam/mul_212training/Adam/add_127*
_output_shapes
:@*
T0
o
training/Adam/sub_127Subconv2d_8/bias/readtraining/Adam/truediv_43*
T0*
_output_shapes
:@
Đ
training/Adam/Assign_123Assigntraining/Adam/Variable_41training/Adam/add_125*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_41
Đ
training/Adam/Assign_124Assigntraining/Adam/Variable_87training/Adam/add_126*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_87*
validate_shape(
¸
training/Adam/Assign_125Assignconv2d_8/biastraining/Adam/sub_127*
use_locking(*
T0* 
_class
loc:@conv2d_8/bias*
validate_shape(*
_output_shapes
:@

training/Adam/mul_213MulAdam/beta_1/readtraining/Adam/Variable_42/read*&
_output_shapes
:@@*
T0
\
training/Adam/sub_128/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_128Subtraining/Adam/sub_128/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ź
training/Adam/mul_214Multraining/Adam/sub_128Ftraining/Adam/gradients/conv2d_9/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
{
training/Adam/add_128Addtraining/Adam/mul_213training/Adam/mul_214*
T0*&
_output_shapes
:@@

training/Adam/mul_215MulAdam/beta_2/readtraining/Adam/Variable_88/read*
T0*&
_output_shapes
:@@
\
training/Adam/sub_129/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_129Subtraining/Adam/sub_129/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_42SquareFtraining/Adam/gradients/conv2d_9/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@@
}
training/Adam/mul_216Multraining/Adam/sub_129training/Adam/Square_42*&
_output_shapes
:@@*
T0
{
training/Adam/add_129Addtraining/Adam/mul_215training/Adam/mul_216*
T0*&
_output_shapes
:@@
y
training/Adam/mul_217Multraining/Adam/mul_2training/Adam/add_128*&
_output_shapes
:@@*
T0
[
training/Adam/Const_86Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_87Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_43/MinimumMinimumtraining/Adam/add_129training/Adam/Const_87*
T0*&
_output_shapes
:@@

training/Adam/clip_by_value_43Maximum&training/Adam/clip_by_value_43/Minimumtraining/Adam/Const_86*&
_output_shapes
:@@*
T0
n
training/Adam/Sqrt_43Sqrttraining/Adam/clip_by_value_43*&
_output_shapes
:@@*
T0
\
training/Adam/add_130/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_130Addtraining/Adam/Sqrt_43training/Adam/add_130/y*&
_output_shapes
:@@*
T0

training/Adam/truediv_44RealDivtraining/Adam/mul_217training/Adam/add_130*
T0*&
_output_shapes
:@@
}
training/Adam/sub_130Subconv2d_9/kernel/readtraining/Adam/truediv_44*&
_output_shapes
:@@*
T0
Ü
training/Adam/Assign_126Assigntraining/Adam/Variable_42training/Adam/add_128*&
_output_shapes
:@@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_42*
validate_shape(
Ü
training/Adam/Assign_127Assigntraining/Adam/Variable_88training/Adam/add_129*&
_output_shapes
:@@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_88*
validate_shape(
Č
training/Adam/Assign_128Assignconv2d_9/kerneltraining/Adam/sub_130*
use_locking(*
T0*"
_class
loc:@conv2d_9/kernel*
validate_shape(*&
_output_shapes
:@@
s
training/Adam/mul_218MulAdam/beta_1/readtraining/Adam/Variable_43/read*
T0*
_output_shapes
:@
\
training/Adam/sub_131/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_131Subtraining/Adam/sub_131/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_219Multraining/Adam/sub_1319training/Adam/gradients/conv2d_9/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
o
training/Adam/add_131Addtraining/Adam/mul_218training/Adam/mul_219*
T0*
_output_shapes
:@
s
training/Adam/mul_220MulAdam/beta_2/readtraining/Adam/Variable_89/read*
T0*
_output_shapes
:@
\
training/Adam/sub_132/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
h
training/Adam/sub_132Subtraining/Adam/sub_132/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_43Square9training/Adam/gradients/conv2d_9/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
q
training/Adam/mul_221Multraining/Adam/sub_132training/Adam/Square_43*
T0*
_output_shapes
:@
o
training/Adam/add_132Addtraining/Adam/mul_220training/Adam/mul_221*
T0*
_output_shapes
:@
m
training/Adam/mul_222Multraining/Adam/mul_2training/Adam/add_131*
_output_shapes
:@*
T0
[
training/Adam/Const_88Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_89Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_44/MinimumMinimumtraining/Adam/add_132training/Adam/Const_89*
_output_shapes
:@*
T0

training/Adam/clip_by_value_44Maximum&training/Adam/clip_by_value_44/Minimumtraining/Adam/Const_88*
T0*
_output_shapes
:@
b
training/Adam/Sqrt_44Sqrttraining/Adam/clip_by_value_44*
T0*
_output_shapes
:@
\
training/Adam/add_133/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
q
training/Adam/add_133Addtraining/Adam/Sqrt_44training/Adam/add_133/y*
T0*
_output_shapes
:@
v
training/Adam/truediv_45RealDivtraining/Adam/mul_222training/Adam/add_133*
T0*
_output_shapes
:@
o
training/Adam/sub_133Subconv2d_9/bias/readtraining/Adam/truediv_45*
T0*
_output_shapes
:@
Đ
training/Adam/Assign_129Assigntraining/Adam/Variable_43training/Adam/add_131*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_43*
validate_shape(
Đ
training/Adam/Assign_130Assigntraining/Adam/Variable_89training/Adam/add_132*,
_class"
 loc:@training/Adam/Variable_89*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
¸
training/Adam/Assign_131Assignconv2d_9/biastraining/Adam/sub_133*
use_locking(*
T0* 
_class
loc:@conv2d_9/bias*
validate_shape(*
_output_shapes
:@

training/Adam/mul_223MulAdam/beta_1/readtraining/Adam/Variable_44/read*
T0*&
_output_shapes
:@
\
training/Adam/sub_134/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_134Subtraining/Adam/sub_134/xAdam/beta_1/read*
T0*
_output_shapes
: 
­
training/Adam/mul_224Multraining/Adam/sub_134Gtraining/Adam/gradients/conv2d_10/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
{
training/Adam/add_134Addtraining/Adam/mul_223training/Adam/mul_224*
T0*&
_output_shapes
:@

training/Adam/mul_225MulAdam/beta_2/readtraining/Adam/Variable_90/read*
T0*&
_output_shapes
:@
\
training/Adam/sub_135/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_135Subtraining/Adam/sub_135/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_44SquareGtraining/Adam/gradients/conv2d_10/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@*
T0
}
training/Adam/mul_226Multraining/Adam/sub_135training/Adam/Square_44*&
_output_shapes
:@*
T0
{
training/Adam/add_135Addtraining/Adam/mul_225training/Adam/mul_226*&
_output_shapes
:@*
T0
y
training/Adam/mul_227Multraining/Adam/mul_2training/Adam/add_134*
T0*&
_output_shapes
:@
[
training/Adam/Const_90Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_91Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_45/MinimumMinimumtraining/Adam/add_135training/Adam/Const_91*&
_output_shapes
:@*
T0

training/Adam/clip_by_value_45Maximum&training/Adam/clip_by_value_45/Minimumtraining/Adam/Const_90*
T0*&
_output_shapes
:@
n
training/Adam/Sqrt_45Sqrttraining/Adam/clip_by_value_45*&
_output_shapes
:@*
T0
\
training/Adam/add_136/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_136Addtraining/Adam/Sqrt_45training/Adam/add_136/y*
T0*&
_output_shapes
:@

training/Adam/truediv_46RealDivtraining/Adam/mul_227training/Adam/add_136*
T0*&
_output_shapes
:@
~
training/Adam/sub_136Subconv2d_10/kernel/readtraining/Adam/truediv_46*&
_output_shapes
:@*
T0
Ü
training/Adam/Assign_132Assigntraining/Adam/Variable_44training/Adam/add_134*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_44*
validate_shape(*&
_output_shapes
:@
Ü
training/Adam/Assign_133Assigntraining/Adam/Variable_90training/Adam/add_135*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_90
Ę
training/Adam/Assign_134Assignconv2d_10/kerneltraining/Adam/sub_136*
use_locking(*
T0*#
_class
loc:@conv2d_10/kernel*
validate_shape(*&
_output_shapes
:@
s
training/Adam/mul_228MulAdam/beta_1/readtraining/Adam/Variable_45/read*
T0*
_output_shapes
:
\
training/Adam/sub_137/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
h
training/Adam/sub_137Subtraining/Adam/sub_137/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_229Multraining/Adam/sub_137:training/Adam/gradients/conv2d_10/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
o
training/Adam/add_137Addtraining/Adam/mul_228training/Adam/mul_229*
T0*
_output_shapes
:
s
training/Adam/mul_230MulAdam/beta_2/readtraining/Adam/Variable_91/read*
T0*
_output_shapes
:
\
training/Adam/sub_138/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_138Subtraining/Adam/sub_138/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_45Square:training/Adam/gradients/conv2d_10/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
q
training/Adam/mul_231Multraining/Adam/sub_138training/Adam/Square_45*
_output_shapes
:*
T0
o
training/Adam/add_138Addtraining/Adam/mul_230training/Adam/mul_231*
_output_shapes
:*
T0
m
training/Adam/mul_232Multraining/Adam/mul_2training/Adam/add_137*
T0*
_output_shapes
:
[
training/Adam/Const_92Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_93Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_46/MinimumMinimumtraining/Adam/add_138training/Adam/Const_93*
T0*
_output_shapes
:

training/Adam/clip_by_value_46Maximum&training/Adam/clip_by_value_46/Minimumtraining/Adam/Const_92*
T0*
_output_shapes
:
b
training/Adam/Sqrt_46Sqrttraining/Adam/clip_by_value_46*
T0*
_output_shapes
:
\
training/Adam/add_139/yConst*
_output_shapes
: *
valueB
 *żÖ3*
dtype0
q
training/Adam/add_139Addtraining/Adam/Sqrt_46training/Adam/add_139/y*
_output_shapes
:*
T0
v
training/Adam/truediv_47RealDivtraining/Adam/mul_232training/Adam/add_139*
T0*
_output_shapes
:
p
training/Adam/sub_139Subconv2d_10/bias/readtraining/Adam/truediv_47*
_output_shapes
:*
T0
Đ
training/Adam/Assign_135Assigntraining/Adam/Variable_45training/Adam/add_137*
T0*,
_class"
 loc:@training/Adam/Variable_45*
validate_shape(*
_output_shapes
:*
use_locking(
Đ
training/Adam/Assign_136Assigntraining/Adam/Variable_91training/Adam/add_138*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_91*
validate_shape(*
_output_shapes
:
ş
training/Adam/Assign_137Assignconv2d_10/biastraining/Adam/sub_139*
use_locking(*
T0*!
_class
loc:@conv2d_10/bias*
validate_shape(*
_output_shapes
:
Ţ
training/group_depsNoOp	^loss/mul^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_100^training/Adam/Assign_101^training/Adam/Assign_102^training/Adam/Assign_103^training/Adam/Assign_104^training/Adam/Assign_105^training/Adam/Assign_106^training/Adam/Assign_107^training/Adam/Assign_108^training/Adam/Assign_109^training/Adam/Assign_11^training/Adam/Assign_110^training/Adam/Assign_111^training/Adam/Assign_112^training/Adam/Assign_113^training/Adam/Assign_114^training/Adam/Assign_115^training/Adam/Assign_116^training/Adam/Assign_117^training/Adam/Assign_118^training/Adam/Assign_119^training/Adam/Assign_12^training/Adam/Assign_120^training/Adam/Assign_121^training/Adam/Assign_122^training/Adam/Assign_123^training/Adam/Assign_124^training/Adam/Assign_125^training/Adam/Assign_126^training/Adam/Assign_127^training/Adam/Assign_128^training/Adam/Assign_129^training/Adam/Assign_13^training/Adam/Assign_130^training/Adam/Assign_131^training/Adam/Assign_132^training/Adam/Assign_133^training/Adam/Assign_134^training/Adam/Assign_135^training/Adam/Assign_136^training/Adam/Assign_137^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_24^training/Adam/Assign_25^training/Adam/Assign_26^training/Adam/Assign_27^training/Adam/Assign_28^training/Adam/Assign_29^training/Adam/Assign_3^training/Adam/Assign_30^training/Adam/Assign_31^training/Adam/Assign_32^training/Adam/Assign_33^training/Adam/Assign_34^training/Adam/Assign_35^training/Adam/Assign_36^training/Adam/Assign_37^training/Adam/Assign_38^training/Adam/Assign_39^training/Adam/Assign_4^training/Adam/Assign_40^training/Adam/Assign_41^training/Adam/Assign_42^training/Adam/Assign_43^training/Adam/Assign_44^training/Adam/Assign_45^training/Adam/Assign_46^training/Adam/Assign_47^training/Adam/Assign_48^training/Adam/Assign_49^training/Adam/Assign_5^training/Adam/Assign_50^training/Adam/Assign_51^training/Adam/Assign_52^training/Adam/Assign_53^training/Adam/Assign_54^training/Adam/Assign_55^training/Adam/Assign_56^training/Adam/Assign_57^training/Adam/Assign_58^training/Adam/Assign_59^training/Adam/Assign_6^training/Adam/Assign_60^training/Adam/Assign_61^training/Adam/Assign_62^training/Adam/Assign_63^training/Adam/Assign_64^training/Adam/Assign_65^training/Adam/Assign_66^training/Adam/Assign_67^training/Adam/Assign_68^training/Adam/Assign_69^training/Adam/Assign_7^training/Adam/Assign_70^training/Adam/Assign_71^training/Adam/Assign_72^training/Adam/Assign_73^training/Adam/Assign_74^training/Adam/Assign_75^training/Adam/Assign_76^training/Adam/Assign_77^training/Adam/Assign_78^training/Adam/Assign_79^training/Adam/Assign_8^training/Adam/Assign_80^training/Adam/Assign_81^training/Adam/Assign_82^training/Adam/Assign_83^training/Adam/Assign_84^training/Adam/Assign_85^training/Adam/Assign_86^training/Adam/Assign_87^training/Adam/Assign_88^training/Adam/Assign_89^training/Adam/Assign_9^training/Adam/Assign_90^training/Adam/Assign_91^training/Adam/Assign_92^training/Adam/Assign_93^training/Adam/Assign_94^training/Adam/Assign_95^training/Adam/Assign_96^training/Adam/Assign_97^training/Adam/Assign_98^training/Adam/Assign_99


group_depsNoOp	^loss/mul

IsVariableInitializedIsVariableInitializedconv1a/kernel*
dtype0*
_output_shapes
: * 
_class
loc:@conv1a/kernel

IsVariableInitialized_1IsVariableInitializedconv1a/bias*
_class
loc:@conv1a/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_2IsVariableInitializedconv1b/kernel*
_output_shapes
: * 
_class
loc:@conv1b/kernel*
dtype0

IsVariableInitialized_3IsVariableInitializedconv1b/bias*
_class
loc:@conv1b/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializedconv2a/kernel*
dtype0*
_output_shapes
: * 
_class
loc:@conv2a/kernel

IsVariableInitialized_5IsVariableInitializedconv2a/bias*
dtype0*
_output_shapes
: *
_class
loc:@conv2a/bias

IsVariableInitialized_6IsVariableInitializedconv2b/kernel* 
_class
loc:@conv2b/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_7IsVariableInitializedconv2b/bias*
dtype0*
_output_shapes
: *
_class
loc:@conv2b/bias

IsVariableInitialized_8IsVariableInitializedconv3a/kernel* 
_class
loc:@conv3a/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_9IsVariableInitializedconv3a/bias*
_class
loc:@conv3a/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitializedconv3b/kernel* 
_class
loc:@conv3b/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_11IsVariableInitializedconv3b/bias*
_class
loc:@conv3b/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_12IsVariableInitializedconv4a/kernel* 
_class
loc:@conv4a/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_13IsVariableInitializedconv4a/bias*
_class
loc:@conv4a/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_14IsVariableInitializedconv4b/kernel* 
_class
loc:@conv4b/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_15IsVariableInitializedconv4b/bias*
dtype0*
_output_shapes
: *
_class
loc:@conv4b/bias

IsVariableInitialized_16IsVariableInitializedconv5a/kernel* 
_class
loc:@conv5a/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_17IsVariableInitializedconv5a/bias*
_class
loc:@conv5a/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_18IsVariableInitializedconv5b/kernel* 
_class
loc:@conv5b/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_19IsVariableInitializedconv5b/bias*
_class
loc:@conv5b/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_20IsVariableInitializedconv6/kernel*
_class
loc:@conv6/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitialized
conv6/bias*
_class
loc:@conv6/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_22IsVariableInitializedconv7a/kernel* 
_class
loc:@conv7a/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_23IsVariableInitializedconv7a/bias*
_class
loc:@conv7a/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_24IsVariableInitializedconv7b/kernel* 
_class
loc:@conv7b/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_25IsVariableInitializedconv7b/bias*
_class
loc:@conv7b/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_26IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_27IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_28IsVariableInitializedconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_29IsVariableInitializedconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_30IsVariableInitializedconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_31IsVariableInitializedconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_32IsVariableInitializedconv2d_4/kernel*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_4/kernel

IsVariableInitialized_33IsVariableInitializedconv2d_4/bias* 
_class
loc:@conv2d_4/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_34IsVariableInitializedconv2d_5/kernel*"
_class
loc:@conv2d_5/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_35IsVariableInitializedconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_36IsVariableInitializedconv2d_6/kernel*"
_class
loc:@conv2d_6/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_37IsVariableInitializedconv2d_6/bias* 
_class
loc:@conv2d_6/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_38IsVariableInitializedconv2d_7/kernel*"
_class
loc:@conv2d_7/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_39IsVariableInitializedconv2d_7/bias* 
_class
loc:@conv2d_7/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_40IsVariableInitializedconv2d_8/kernel*"
_class
loc:@conv2d_8/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_41IsVariableInitializedconv2d_8/bias* 
_class
loc:@conv2d_8/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_42IsVariableInitializedconv2d_9/kernel*
_output_shapes
: *"
_class
loc:@conv2d_9/kernel*
dtype0

IsVariableInitialized_43IsVariableInitializedconv2d_9/bias* 
_class
loc:@conv2d_9/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_44IsVariableInitializedconv2d_10/kernel*#
_class
loc:@conv2d_10/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_45IsVariableInitializedconv2d_10/bias*!
_class
loc:@conv2d_10/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_46IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
{
IsVariableInitialized_47IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 

IsVariableInitialized_48IsVariableInitializedAdam/beta_1*
dtype0*
_output_shapes
: *
_class
loc:@Adam/beta_1

IsVariableInitialized_49IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_50IsVariableInitialized
Adam/decay*
_output_shapes
: *
_class
loc:@Adam/decay*
dtype0

IsVariableInitialized_51IsVariableInitializedtraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
dtype0*
_output_shapes
: 

IsVariableInitialized_52IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_53IsVariableInitializedtraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_54IsVariableInitializedtraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0*
_output_shapes
: 

IsVariableInitialized_55IsVariableInitializedtraining/Adam/Variable_4*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_4*
dtype0

IsVariableInitialized_56IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
: 

IsVariableInitialized_57IsVariableInitializedtraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes
: 

IsVariableInitialized_58IsVariableInitializedtraining/Adam/Variable_7*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_7

IsVariableInitialized_59IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes
: 

IsVariableInitialized_60IsVariableInitializedtraining/Adam/Variable_9*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_9*
dtype0

IsVariableInitialized_61IsVariableInitializedtraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0*
_output_shapes
: 

IsVariableInitialized_62IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
: 

IsVariableInitialized_63IsVariableInitializedtraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
: 

IsVariableInitialized_64IsVariableInitializedtraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
dtype0*
_output_shapes
: 

IsVariableInitialized_65IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes
: 

IsVariableInitialized_66IsVariableInitializedtraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
dtype0*
_output_shapes
: 

IsVariableInitialized_67IsVariableInitializedtraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0*
_output_shapes
: 

IsVariableInitialized_68IsVariableInitializedtraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
dtype0*
_output_shapes
: 

IsVariableInitialized_69IsVariableInitializedtraining/Adam/Variable_18*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_18

IsVariableInitialized_70IsVariableInitializedtraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
dtype0*
_output_shapes
: 

IsVariableInitialized_71IsVariableInitializedtraining/Adam/Variable_20*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_20

IsVariableInitialized_72IsVariableInitializedtraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
: 

IsVariableInitialized_73IsVariableInitializedtraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
dtype0*
_output_shapes
: 

IsVariableInitialized_74IsVariableInitializedtraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
dtype0*
_output_shapes
: 

IsVariableInitialized_75IsVariableInitializedtraining/Adam/Variable_24*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_24*
dtype0

IsVariableInitialized_76IsVariableInitializedtraining/Adam/Variable_25*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_25

IsVariableInitialized_77IsVariableInitializedtraining/Adam/Variable_26*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_26

IsVariableInitialized_78IsVariableInitializedtraining/Adam/Variable_27*,
_class"
 loc:@training/Adam/Variable_27*
dtype0*
_output_shapes
: 

IsVariableInitialized_79IsVariableInitializedtraining/Adam/Variable_28*,
_class"
 loc:@training/Adam/Variable_28*
dtype0*
_output_shapes
: 

IsVariableInitialized_80IsVariableInitializedtraining/Adam/Variable_29*,
_class"
 loc:@training/Adam/Variable_29*
dtype0*
_output_shapes
: 

IsVariableInitialized_81IsVariableInitializedtraining/Adam/Variable_30*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_30*
dtype0

IsVariableInitialized_82IsVariableInitializedtraining/Adam/Variable_31*,
_class"
 loc:@training/Adam/Variable_31*
dtype0*
_output_shapes
: 

IsVariableInitialized_83IsVariableInitializedtraining/Adam/Variable_32*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_32

IsVariableInitialized_84IsVariableInitializedtraining/Adam/Variable_33*,
_class"
 loc:@training/Adam/Variable_33*
dtype0*
_output_shapes
: 

IsVariableInitialized_85IsVariableInitializedtraining/Adam/Variable_34*,
_class"
 loc:@training/Adam/Variable_34*
dtype0*
_output_shapes
: 

IsVariableInitialized_86IsVariableInitializedtraining/Adam/Variable_35*,
_class"
 loc:@training/Adam/Variable_35*
dtype0*
_output_shapes
: 

IsVariableInitialized_87IsVariableInitializedtraining/Adam/Variable_36*,
_class"
 loc:@training/Adam/Variable_36*
dtype0*
_output_shapes
: 

IsVariableInitialized_88IsVariableInitializedtraining/Adam/Variable_37*,
_class"
 loc:@training/Adam/Variable_37*
dtype0*
_output_shapes
: 

IsVariableInitialized_89IsVariableInitializedtraining/Adam/Variable_38*,
_class"
 loc:@training/Adam/Variable_38*
dtype0*
_output_shapes
: 

IsVariableInitialized_90IsVariableInitializedtraining/Adam/Variable_39*,
_class"
 loc:@training/Adam/Variable_39*
dtype0*
_output_shapes
: 

IsVariableInitialized_91IsVariableInitializedtraining/Adam/Variable_40*,
_class"
 loc:@training/Adam/Variable_40*
dtype0*
_output_shapes
: 

IsVariableInitialized_92IsVariableInitializedtraining/Adam/Variable_41*,
_class"
 loc:@training/Adam/Variable_41*
dtype0*
_output_shapes
: 

IsVariableInitialized_93IsVariableInitializedtraining/Adam/Variable_42*,
_class"
 loc:@training/Adam/Variable_42*
dtype0*
_output_shapes
: 

IsVariableInitialized_94IsVariableInitializedtraining/Adam/Variable_43*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_43*
dtype0

IsVariableInitialized_95IsVariableInitializedtraining/Adam/Variable_44*,
_class"
 loc:@training/Adam/Variable_44*
dtype0*
_output_shapes
: 

IsVariableInitialized_96IsVariableInitializedtraining/Adam/Variable_45*,
_class"
 loc:@training/Adam/Variable_45*
dtype0*
_output_shapes
: 

IsVariableInitialized_97IsVariableInitializedtraining/Adam/Variable_46*,
_class"
 loc:@training/Adam/Variable_46*
dtype0*
_output_shapes
: 

IsVariableInitialized_98IsVariableInitializedtraining/Adam/Variable_47*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_47

IsVariableInitialized_99IsVariableInitializedtraining/Adam/Variable_48*,
_class"
 loc:@training/Adam/Variable_48*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_100IsVariableInitializedtraining/Adam/Variable_49*,
_class"
 loc:@training/Adam/Variable_49*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_101IsVariableInitializedtraining/Adam/Variable_50*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_50*
dtype0
 
IsVariableInitialized_102IsVariableInitializedtraining/Adam/Variable_51*,
_class"
 loc:@training/Adam/Variable_51*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_103IsVariableInitializedtraining/Adam/Variable_52*,
_class"
 loc:@training/Adam/Variable_52*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_104IsVariableInitializedtraining/Adam/Variable_53*,
_class"
 loc:@training/Adam/Variable_53*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_105IsVariableInitializedtraining/Adam/Variable_54*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_54*
dtype0
 
IsVariableInitialized_106IsVariableInitializedtraining/Adam/Variable_55*,
_class"
 loc:@training/Adam/Variable_55*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_107IsVariableInitializedtraining/Adam/Variable_56*,
_class"
 loc:@training/Adam/Variable_56*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_108IsVariableInitializedtraining/Adam/Variable_57*,
_class"
 loc:@training/Adam/Variable_57*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_109IsVariableInitializedtraining/Adam/Variable_58*,
_class"
 loc:@training/Adam/Variable_58*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_110IsVariableInitializedtraining/Adam/Variable_59*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_59*
dtype0
 
IsVariableInitialized_111IsVariableInitializedtraining/Adam/Variable_60*,
_class"
 loc:@training/Adam/Variable_60*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_112IsVariableInitializedtraining/Adam/Variable_61*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_61
 
IsVariableInitialized_113IsVariableInitializedtraining/Adam/Variable_62*,
_class"
 loc:@training/Adam/Variable_62*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_114IsVariableInitializedtraining/Adam/Variable_63*,
_class"
 loc:@training/Adam/Variable_63*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_115IsVariableInitializedtraining/Adam/Variable_64*,
_class"
 loc:@training/Adam/Variable_64*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_116IsVariableInitializedtraining/Adam/Variable_65*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_65*
dtype0
 
IsVariableInitialized_117IsVariableInitializedtraining/Adam/Variable_66*,
_class"
 loc:@training/Adam/Variable_66*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_118IsVariableInitializedtraining/Adam/Variable_67*,
_class"
 loc:@training/Adam/Variable_67*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_119IsVariableInitializedtraining/Adam/Variable_68*,
_class"
 loc:@training/Adam/Variable_68*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_120IsVariableInitializedtraining/Adam/Variable_69*,
_class"
 loc:@training/Adam/Variable_69*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_121IsVariableInitializedtraining/Adam/Variable_70*,
_class"
 loc:@training/Adam/Variable_70*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_122IsVariableInitializedtraining/Adam/Variable_71*,
_class"
 loc:@training/Adam/Variable_71*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_123IsVariableInitializedtraining/Adam/Variable_72*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_72*
dtype0
 
IsVariableInitialized_124IsVariableInitializedtraining/Adam/Variable_73*,
_class"
 loc:@training/Adam/Variable_73*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_125IsVariableInitializedtraining/Adam/Variable_74*,
_class"
 loc:@training/Adam/Variable_74*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_126IsVariableInitializedtraining/Adam/Variable_75*,
_class"
 loc:@training/Adam/Variable_75*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_127IsVariableInitializedtraining/Adam/Variable_76*,
_class"
 loc:@training/Adam/Variable_76*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_128IsVariableInitializedtraining/Adam/Variable_77*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_77*
dtype0
 
IsVariableInitialized_129IsVariableInitializedtraining/Adam/Variable_78*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_78*
dtype0
 
IsVariableInitialized_130IsVariableInitializedtraining/Adam/Variable_79*,
_class"
 loc:@training/Adam/Variable_79*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_131IsVariableInitializedtraining/Adam/Variable_80*,
_class"
 loc:@training/Adam/Variable_80*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_132IsVariableInitializedtraining/Adam/Variable_81*,
_class"
 loc:@training/Adam/Variable_81*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_133IsVariableInitializedtraining/Adam/Variable_82*,
_class"
 loc:@training/Adam/Variable_82*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_134IsVariableInitializedtraining/Adam/Variable_83*,
_class"
 loc:@training/Adam/Variable_83*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_135IsVariableInitializedtraining/Adam/Variable_84*,
_class"
 loc:@training/Adam/Variable_84*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_136IsVariableInitializedtraining/Adam/Variable_85*,
_class"
 loc:@training/Adam/Variable_85*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_137IsVariableInitializedtraining/Adam/Variable_86*,
_class"
 loc:@training/Adam/Variable_86*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_138IsVariableInitializedtraining/Adam/Variable_87*,
_class"
 loc:@training/Adam/Variable_87*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_139IsVariableInitializedtraining/Adam/Variable_88*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_88*
dtype0
 
IsVariableInitialized_140IsVariableInitializedtraining/Adam/Variable_89*,
_class"
 loc:@training/Adam/Variable_89*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_141IsVariableInitializedtraining/Adam/Variable_90*,
_class"
 loc:@training/Adam/Variable_90*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_142IsVariableInitializedtraining/Adam/Variable_91*,
_class"
 loc:@training/Adam/Variable_91*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_143IsVariableInitializedtraining/Adam/Variable_92*,
_class"
 loc:@training/Adam/Variable_92*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_144IsVariableInitializedtraining/Adam/Variable_93*,
_class"
 loc:@training/Adam/Variable_93*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_145IsVariableInitializedtraining/Adam/Variable_94*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_94
 
IsVariableInitialized_146IsVariableInitializedtraining/Adam/Variable_95*,
_class"
 loc:@training/Adam/Variable_95*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_147IsVariableInitializedtraining/Adam/Variable_96*,
_class"
 loc:@training/Adam/Variable_96*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_148IsVariableInitializedtraining/Adam/Variable_97*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_97
 
IsVariableInitialized_149IsVariableInitializedtraining/Adam/Variable_98*,
_class"
 loc:@training/Adam/Variable_98*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_150IsVariableInitializedtraining/Adam/Variable_99*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_99
˘
IsVariableInitialized_151IsVariableInitializedtraining/Adam/Variable_100*-
_class#
!loc:@training/Adam/Variable_100*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_152IsVariableInitializedtraining/Adam/Variable_101*
dtype0*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_101
˘
IsVariableInitialized_153IsVariableInitializedtraining/Adam/Variable_102*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_102*
dtype0
˘
IsVariableInitialized_154IsVariableInitializedtraining/Adam/Variable_103*-
_class#
!loc:@training/Adam/Variable_103*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_155IsVariableInitializedtraining/Adam/Variable_104*-
_class#
!loc:@training/Adam/Variable_104*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_156IsVariableInitializedtraining/Adam/Variable_105*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_105*
dtype0
˘
IsVariableInitialized_157IsVariableInitializedtraining/Adam/Variable_106*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_106*
dtype0
˘
IsVariableInitialized_158IsVariableInitializedtraining/Adam/Variable_107*-
_class#
!loc:@training/Adam/Variable_107*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_159IsVariableInitializedtraining/Adam/Variable_108*-
_class#
!loc:@training/Adam/Variable_108*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_160IsVariableInitializedtraining/Adam/Variable_109*-
_class#
!loc:@training/Adam/Variable_109*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_161IsVariableInitializedtraining/Adam/Variable_110*-
_class#
!loc:@training/Adam/Variable_110*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_162IsVariableInitializedtraining/Adam/Variable_111*
dtype0*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_111
˘
IsVariableInitialized_163IsVariableInitializedtraining/Adam/Variable_112*-
_class#
!loc:@training/Adam/Variable_112*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_164IsVariableInitializedtraining/Adam/Variable_113*-
_class#
!loc:@training/Adam/Variable_113*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_165IsVariableInitializedtraining/Adam/Variable_114*-
_class#
!loc:@training/Adam/Variable_114*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_166IsVariableInitializedtraining/Adam/Variable_115*
dtype0*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_115
˘
IsVariableInitialized_167IsVariableInitializedtraining/Adam/Variable_116*-
_class#
!loc:@training/Adam/Variable_116*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_168IsVariableInitializedtraining/Adam/Variable_117*-
_class#
!loc:@training/Adam/Variable_117*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_169IsVariableInitializedtraining/Adam/Variable_118*-
_class#
!loc:@training/Adam/Variable_118*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_170IsVariableInitializedtraining/Adam/Variable_119*-
_class#
!loc:@training/Adam/Variable_119*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_171IsVariableInitializedtraining/Adam/Variable_120*
dtype0*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_120
˘
IsVariableInitialized_172IsVariableInitializedtraining/Adam/Variable_121*-
_class#
!loc:@training/Adam/Variable_121*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_173IsVariableInitializedtraining/Adam/Variable_122*-
_class#
!loc:@training/Adam/Variable_122*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_174IsVariableInitializedtraining/Adam/Variable_123*-
_class#
!loc:@training/Adam/Variable_123*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_175IsVariableInitializedtraining/Adam/Variable_124*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_124*
dtype0
˘
IsVariableInitialized_176IsVariableInitializedtraining/Adam/Variable_125*-
_class#
!loc:@training/Adam/Variable_125*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_177IsVariableInitializedtraining/Adam/Variable_126*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_126*
dtype0
˘
IsVariableInitialized_178IsVariableInitializedtraining/Adam/Variable_127*-
_class#
!loc:@training/Adam/Variable_127*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_179IsVariableInitializedtraining/Adam/Variable_128*-
_class#
!loc:@training/Adam/Variable_128*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_180IsVariableInitializedtraining/Adam/Variable_129*-
_class#
!loc:@training/Adam/Variable_129*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_181IsVariableInitializedtraining/Adam/Variable_130*-
_class#
!loc:@training/Adam/Variable_130*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_182IsVariableInitializedtraining/Adam/Variable_131*-
_class#
!loc:@training/Adam/Variable_131*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_183IsVariableInitializedtraining/Adam/Variable_132*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_132*
dtype0
˘
IsVariableInitialized_184IsVariableInitializedtraining/Adam/Variable_133*-
_class#
!loc:@training/Adam/Variable_133*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_185IsVariableInitializedtraining/Adam/Variable_134*-
_class#
!loc:@training/Adam/Variable_134*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_186IsVariableInitializedtraining/Adam/Variable_135*-
_class#
!loc:@training/Adam/Variable_135*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_187IsVariableInitializedtraining/Adam/Variable_136*-
_class#
!loc:@training/Adam/Variable_136*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_188IsVariableInitializedtraining/Adam/Variable_137*-
_class#
!loc:@training/Adam/Variable_137*
dtype0*
_output_shapes
: 
/
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^conv1a/bias/Assign^conv1a/kernel/Assign^conv1b/bias/Assign^conv1b/kernel/Assign^conv2a/bias/Assign^conv2a/kernel/Assign^conv2b/bias/Assign^conv2b/kernel/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_10/bias/Assign^conv2d_10/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^conv2d_3/bias/Assign^conv2d_3/kernel/Assign^conv2d_4/bias/Assign^conv2d_4/kernel/Assign^conv2d_5/bias/Assign^conv2d_5/kernel/Assign^conv2d_6/bias/Assign^conv2d_6/kernel/Assign^conv2d_7/bias/Assign^conv2d_7/kernel/Assign^conv2d_8/bias/Assign^conv2d_8/kernel/Assign^conv2d_9/bias/Assign^conv2d_9/kernel/Assign^conv3a/bias/Assign^conv3a/kernel/Assign^conv3b/bias/Assign^conv3b/kernel/Assign^conv4a/bias/Assign^conv4a/kernel/Assign^conv4b/bias/Assign^conv4b/kernel/Assign^conv5a/bias/Assign^conv5a/kernel/Assign^conv5b/bias/Assign^conv5b/kernel/Assign^conv6/bias/Assign^conv6/kernel/Assign^conv7a/bias/Assign^conv7a/kernel/Assign^conv7b/bias/Assign^conv7b/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign"^training/Adam/Variable_100/Assign"^training/Adam/Variable_101/Assign"^training/Adam/Variable_102/Assign"^training/Adam/Variable_103/Assign"^training/Adam/Variable_104/Assign"^training/Adam/Variable_105/Assign"^training/Adam/Variable_106/Assign"^training/Adam/Variable_107/Assign"^training/Adam/Variable_108/Assign"^training/Adam/Variable_109/Assign!^training/Adam/Variable_11/Assign"^training/Adam/Variable_110/Assign"^training/Adam/Variable_111/Assign"^training/Adam/Variable_112/Assign"^training/Adam/Variable_113/Assign"^training/Adam/Variable_114/Assign"^training/Adam/Variable_115/Assign"^training/Adam/Variable_116/Assign"^training/Adam/Variable_117/Assign"^training/Adam/Variable_118/Assign"^training/Adam/Variable_119/Assign!^training/Adam/Variable_12/Assign"^training/Adam/Variable_120/Assign"^training/Adam/Variable_121/Assign"^training/Adam/Variable_122/Assign"^training/Adam/Variable_123/Assign"^training/Adam/Variable_124/Assign"^training/Adam/Variable_125/Assign"^training/Adam/Variable_126/Assign"^training/Adam/Variable_127/Assign"^training/Adam/Variable_128/Assign"^training/Adam/Variable_129/Assign!^training/Adam/Variable_13/Assign"^training/Adam/Variable_130/Assign"^training/Adam/Variable_131/Assign"^training/Adam/Variable_132/Assign"^training/Adam/Variable_133/Assign"^training/Adam/Variable_134/Assign"^training/Adam/Variable_135/Assign"^training/Adam/Variable_136/Assign"^training/Adam/Variable_137/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign!^training/Adam/Variable_24/Assign!^training/Adam/Variable_25/Assign!^training/Adam/Variable_26/Assign!^training/Adam/Variable_27/Assign!^training/Adam/Variable_28/Assign!^training/Adam/Variable_29/Assign ^training/Adam/Variable_3/Assign!^training/Adam/Variable_30/Assign!^training/Adam/Variable_31/Assign!^training/Adam/Variable_32/Assign!^training/Adam/Variable_33/Assign!^training/Adam/Variable_34/Assign!^training/Adam/Variable_35/Assign!^training/Adam/Variable_36/Assign!^training/Adam/Variable_37/Assign!^training/Adam/Variable_38/Assign!^training/Adam/Variable_39/Assign ^training/Adam/Variable_4/Assign!^training/Adam/Variable_40/Assign!^training/Adam/Variable_41/Assign!^training/Adam/Variable_42/Assign!^training/Adam/Variable_43/Assign!^training/Adam/Variable_44/Assign!^training/Adam/Variable_45/Assign!^training/Adam/Variable_46/Assign!^training/Adam/Variable_47/Assign!^training/Adam/Variable_48/Assign!^training/Adam/Variable_49/Assign ^training/Adam/Variable_5/Assign!^training/Adam/Variable_50/Assign!^training/Adam/Variable_51/Assign!^training/Adam/Variable_52/Assign!^training/Adam/Variable_53/Assign!^training/Adam/Variable_54/Assign!^training/Adam/Variable_55/Assign!^training/Adam/Variable_56/Assign!^training/Adam/Variable_57/Assign!^training/Adam/Variable_58/Assign!^training/Adam/Variable_59/Assign ^training/Adam/Variable_6/Assign!^training/Adam/Variable_60/Assign!^training/Adam/Variable_61/Assign!^training/Adam/Variable_62/Assign!^training/Adam/Variable_63/Assign!^training/Adam/Variable_64/Assign!^training/Adam/Variable_65/Assign!^training/Adam/Variable_66/Assign!^training/Adam/Variable_67/Assign!^training/Adam/Variable_68/Assign!^training/Adam/Variable_69/Assign ^training/Adam/Variable_7/Assign!^training/Adam/Variable_70/Assign!^training/Adam/Variable_71/Assign!^training/Adam/Variable_72/Assign!^training/Adam/Variable_73/Assign!^training/Adam/Variable_74/Assign!^training/Adam/Variable_75/Assign!^training/Adam/Variable_76/Assign!^training/Adam/Variable_77/Assign!^training/Adam/Variable_78/Assign!^training/Adam/Variable_79/Assign ^training/Adam/Variable_8/Assign!^training/Adam/Variable_80/Assign!^training/Adam/Variable_81/Assign!^training/Adam/Variable_82/Assign!^training/Adam/Variable_83/Assign!^training/Adam/Variable_84/Assign!^training/Adam/Variable_85/Assign!^training/Adam/Variable_86/Assign!^training/Adam/Variable_87/Assign!^training/Adam/Variable_88/Assign!^training/Adam/Variable_89/Assign ^training/Adam/Variable_9/Assign!^training/Adam/Variable_90/Assign!^training/Adam/Variable_91/Assign!^training/Adam/Variable_92/Assign!^training/Adam/Variable_93/Assign!^training/Adam/Variable_94/Assign!^training/Adam/Variable_95/Assign!^training/Adam/Variable_96/Assign!^training/Adam/Variable_97/Assign!^training/Adam/Variable_98/Assign!^training/Adam/Variable_99/Assign"Ş?ł&ű     ĂÄř	Ď,¤tV$×AJö
đ/Î/
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
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
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
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
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
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
/
Sign
x"T
y"T"
Ttype:

2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
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

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *1.13.12v1.13.1-0-g6612da8951ĽÜ
~
input_1Placeholder*&
shape:˙˙˙˙˙˙˙˙˙¸Đ*
dtype0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
v
conv1a/truncated_normal/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
conv1a/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv1a/truncated_normal/stddevConst*
valueB
 *k>*
dtype0*
_output_shapes
: 
ľ
'conv1a/truncated_normal/TruncatedNormalTruncatedNormalconv1a/truncated_normal/shape*
T0*
dtype0*
seed2Ş*&
_output_shapes
:@*
seedą˙ĺ)

conv1a/truncated_normal/mulMul'conv1a/truncated_normal/TruncatedNormalconv1a/truncated_normal/stddev*
T0*&
_output_shapes
:@

conv1a/truncated_normalAddconv1a/truncated_normal/mulconv1a/truncated_normal/mean*&
_output_shapes
:@*
T0

conv1a/kernel
VariableV2*
	container *&
_output_shapes
:@*
shape:@*
shared_name *
dtype0
Â
conv1a/kernel/AssignAssignconv1a/kernelconv1a/truncated_normal*
use_locking(*
T0* 
_class
loc:@conv1a/kernel*
validate_shape(*&
_output_shapes
:@

conv1a/kernel/readIdentityconv1a/kernel*&
_output_shapes
:@*
T0* 
_class
loc:@conv1a/kernel
Y
conv1a/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
w
conv1a/bias
VariableV2*
shape:@*
shared_name *
dtype0*
	container *
_output_shapes
:@
Ľ
conv1a/bias/AssignAssignconv1a/biasconv1a/Const*
_class
loc:@conv1a/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
n
conv1a/bias/readIdentityconv1a/bias*
T0*
_class
loc:@conv1a/bias*
_output_shapes
:@
q
 conv1a/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
ă
conv1a/convolutionConv2Dinput_1conv1a/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
	dilations


conv1a/BiasAddBiasAddconv1a/convolutionconv1a/bias/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0*
data_formatNHWC
_
conv1a/ReluReluconv1a/BiasAdd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
v
conv1b/truncated_normal/shapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
a
conv1b/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv1b/truncated_normal/stddevConst*
valueB
 *¸1=*
dtype0*
_output_shapes
: 
ś
'conv1b/truncated_normal/TruncatedNormalTruncatedNormalconv1b/truncated_normal/shape*
T0*
dtype0*
seed2˛ś*&
_output_shapes
:@@*
seedą˙ĺ)

conv1b/truncated_normal/mulMul'conv1b/truncated_normal/TruncatedNormalconv1b/truncated_normal/stddev*&
_output_shapes
:@@*
T0

conv1b/truncated_normalAddconv1b/truncated_normal/mulconv1b/truncated_normal/mean*
T0*&
_output_shapes
:@@

conv1b/kernel
VariableV2*
shared_name *
dtype0*
	container *&
_output_shapes
:@@*
shape:@@
Â
conv1b/kernel/AssignAssignconv1b/kernelconv1b/truncated_normal*
T0* 
_class
loc:@conv1b/kernel*
validate_shape(*&
_output_shapes
:@@*
use_locking(

conv1b/kernel/readIdentityconv1b/kernel*
T0* 
_class
loc:@conv1b/kernel*&
_output_shapes
:@@
Y
conv1b/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
w
conv1b/bias
VariableV2*
shape:@*
shared_name *
dtype0*
	container *
_output_shapes
:@
Ľ
conv1b/bias/AssignAssignconv1b/biasconv1b/Const*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv1b/bias*
validate_shape(
n
conv1b/bias/readIdentityconv1b/bias*
_output_shapes
:@*
T0*
_class
loc:@conv1b/bias
q
 conv1b/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
ç
conv1b/convolutionConv2Dconv1a/Reluconv1b/kernel/read*
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

conv1b/BiasAddBiasAddconv1b/convolutionconv1b/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
_
conv1b/ReluReluconv1b/BiasAdd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
´
pool1/MaxPoolMaxPoolconv1b/Relu*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@
v
conv2a/truncated_normal/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
a
conv2a/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv2a/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *¸1=
ˇ
'conv2a/truncated_normal/TruncatedNormalTruncatedNormalconv2a/truncated_normal/shape*
dtype0*
seed2żŞ*'
_output_shapes
:@*
seedą˙ĺ)*
T0

conv2a/truncated_normal/mulMul'conv2a/truncated_normal/TruncatedNormalconv2a/truncated_normal/stddev*'
_output_shapes
:@*
T0

conv2a/truncated_normalAddconv2a/truncated_normal/mulconv2a/truncated_normal/mean*
T0*'
_output_shapes
:@

conv2a/kernel
VariableV2*
	container *'
_output_shapes
:@*
shape:@*
shared_name *
dtype0
Ă
conv2a/kernel/AssignAssignconv2a/kernelconv2a/truncated_normal*'
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@conv2a/kernel*
validate_shape(

conv2a/kernel/readIdentityconv2a/kernel*
T0* 
_class
loc:@conv2a/kernel*'
_output_shapes
:@
[
conv2a/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
y
conv2a/bias
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ś
conv2a/bias/AssignAssignconv2a/biasconv2a/Const*
use_locking(*
T0*
_class
loc:@conv2a/bias*
validate_shape(*
_output_shapes	
:
o
conv2a/bias/readIdentityconv2a/bias*
T0*
_class
loc:@conv2a/bias*
_output_shapes	
:
q
 conv2a/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ę
conv2a/convolutionConv2Dpool1/MaxPoolconv2a/kernel/read*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv2a/BiasAddBiasAddconv2a/convolutionconv2a/bias/read*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0
`
conv2a/ReluReluconv2a/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
v
conv2b/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
a
conv2b/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv2b/truncated_normal/stddevConst*
valueB
 *B=*
dtype0*
_output_shapes
: 
¸
'conv2b/truncated_normal/TruncatedNormalTruncatedNormalconv2b/truncated_normal/shape*
dtype0*
seed2éŮŔ*(
_output_shapes
:*
seedą˙ĺ)*
T0

conv2b/truncated_normal/mulMul'conv2b/truncated_normal/TruncatedNormalconv2b/truncated_normal/stddev*
T0*(
_output_shapes
:

conv2b/truncated_normalAddconv2b/truncated_normal/mulconv2b/truncated_normal/mean*
T0*(
_output_shapes
:

conv2b/kernel
VariableV2*
shape:*
shared_name *
dtype0*
	container *(
_output_shapes
:
Ä
conv2b/kernel/AssignAssignconv2b/kernelconv2b/truncated_normal*
use_locking(*
T0* 
_class
loc:@conv2b/kernel*
validate_shape(*(
_output_shapes
:

conv2b/kernel/readIdentityconv2b/kernel*(
_output_shapes
:*
T0* 
_class
loc:@conv2b/kernel
[
conv2b/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
y
conv2b/bias
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ś
conv2b/bias/AssignAssignconv2b/biasconv2b/Const*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv2b/bias
o
conv2b/bias/readIdentityconv2b/bias*
_output_shapes	
:*
T0*
_class
loc:@conv2b/bias
q
 conv2b/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
č
conv2b/convolutionConv2Dconv2a/Reluconv2b/kernel/read*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

conv2b/BiasAddBiasAddconv2b/convolutionconv2b/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
`
conv2b/ReluReluconv2b/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
ľ
pool2/MaxPoolMaxPoolconv2b/Relu*
ksize
*
paddingVALID*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
data_formatNHWC
v
conv3a/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
a
conv3a/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv3a/truncated_normal/stddevConst*
valueB
 *B=*
dtype0*
_output_shapes
: 
¸
'conv3a/truncated_normal/TruncatedNormalTruncatedNormalconv3a/truncated_normal/shape*
dtype0*
seed2ŤíŻ*(
_output_shapes
:*
seedą˙ĺ)*
T0

conv3a/truncated_normal/mulMul'conv3a/truncated_normal/TruncatedNormalconv3a/truncated_normal/stddev*
T0*(
_output_shapes
:

conv3a/truncated_normalAddconv3a/truncated_normal/mulconv3a/truncated_normal/mean*(
_output_shapes
:*
T0

conv3a/kernel
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
Ä
conv3a/kernel/AssignAssignconv3a/kernelconv3a/truncated_normal*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv3a/kernel

conv3a/kernel/readIdentityconv3a/kernel*
T0* 
_class
loc:@conv3a/kernel*(
_output_shapes
:
[
conv3a/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
y
conv3a/bias
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ś
conv3a/bias/AssignAssignconv3a/biasconv3a/Const*
use_locking(*
T0*
_class
loc:@conv3a/bias*
validate_shape(*
_output_shapes	
:
o
conv3a/bias/readIdentityconv3a/bias*
T0*
_class
loc:@conv3a/bias*
_output_shapes	
:
q
 conv3a/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ę
conv3a/convolutionConv2Dpool2/MaxPoolconv3a/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙

conv3a/BiasAddBiasAddconv3a/convolutionconv3a/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
`
conv3a/ReluReluconv3a/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
v
conv3b/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
a
conv3b/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv3b/truncated_normal/stddevConst*
valueB
 *¸1	=*
dtype0*
_output_shapes
: 
¸
'conv3b/truncated_normal/TruncatedNormalTruncatedNormalconv3b/truncated_normal/shape*
seedą˙ĺ)*
T0*
dtype0*
seed2Ůá*(
_output_shapes
:

conv3b/truncated_normal/mulMul'conv3b/truncated_normal/TruncatedNormalconv3b/truncated_normal/stddev*
T0*(
_output_shapes
:

conv3b/truncated_normalAddconv3b/truncated_normal/mulconv3b/truncated_normal/mean*
T0*(
_output_shapes
:

conv3b/kernel
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
Ä
conv3b/kernel/AssignAssignconv3b/kernelconv3b/truncated_normal*
use_locking(*
T0* 
_class
loc:@conv3b/kernel*
validate_shape(*(
_output_shapes
:

conv3b/kernel/readIdentityconv3b/kernel*
T0* 
_class
loc:@conv3b/kernel*(
_output_shapes
:
[
conv3b/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
y
conv3b/bias
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ś
conv3b/bias/AssignAssignconv3b/biasconv3b/Const*
use_locking(*
T0*
_class
loc:@conv3b/bias*
validate_shape(*
_output_shapes	
:
o
conv3b/bias/readIdentityconv3b/bias*
T0*
_class
loc:@conv3b/bias*
_output_shapes	
:
q
 conv3b/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
č
conv3b/convolutionConv2Dconv3a/Reluconv3b/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
	dilations


conv3b/BiasAddBiasAddconv3b/convolutionconv3b/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
`
conv3b/ReluReluconv3b/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
ľ
pool3/MaxPoolMaxPoolconv3b/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
v
conv4a/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
a
conv4a/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv4a/truncated_normal/stddevConst*
valueB
 *¸1	=*
dtype0*
_output_shapes
: 
ˇ
'conv4a/truncated_normal/TruncatedNormalTruncatedNormalconv4a/truncated_normal/shape*
T0*
dtype0*
seed2żŔX*(
_output_shapes
:*
seedą˙ĺ)

conv4a/truncated_normal/mulMul'conv4a/truncated_normal/TruncatedNormalconv4a/truncated_normal/stddev*
T0*(
_output_shapes
:

conv4a/truncated_normalAddconv4a/truncated_normal/mulconv4a/truncated_normal/mean*
T0*(
_output_shapes
:

conv4a/kernel
VariableV2*
	container *(
_output_shapes
:*
shape:*
shared_name *
dtype0
Ä
conv4a/kernel/AssignAssignconv4a/kernelconv4a/truncated_normal*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv4a/kernel

conv4a/kernel/readIdentityconv4a/kernel* 
_class
loc:@conv4a/kernel*(
_output_shapes
:*
T0
[
conv4a/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
y
conv4a/bias
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ś
conv4a/bias/AssignAssignconv4a/biasconv4a/Const*
use_locking(*
T0*
_class
loc:@conv4a/bias*
validate_shape(*
_output_shapes	
:
o
conv4a/bias/readIdentityconv4a/bias*
T0*
_class
loc:@conv4a/bias*
_output_shapes	
:
q
 conv4a/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
ę
conv4a/convolutionConv2Dpool3/MaxPoolconv4a/kernel/read*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv4a/BiasAddBiasAddconv4a/convolutionconv4a/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
`
conv4a/ReluReluconv4a/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
v
conv4b/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
a
conv4b/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv4b/truncated_normal/stddevConst*
valueB
 *Â<*
dtype0*
_output_shapes
: 
¸
'conv4b/truncated_normal/TruncatedNormalTruncatedNormalconv4b/truncated_normal/shape*
dtype0*
seed2ÜŠÇ*(
_output_shapes
:*
seedą˙ĺ)*
T0

conv4b/truncated_normal/mulMul'conv4b/truncated_normal/TruncatedNormalconv4b/truncated_normal/stddev*
T0*(
_output_shapes
:

conv4b/truncated_normalAddconv4b/truncated_normal/mulconv4b/truncated_normal/mean*
T0*(
_output_shapes
:

conv4b/kernel
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:*
shape:
Ä
conv4b/kernel/AssignAssignconv4b/kernelconv4b/truncated_normal*
use_locking(*
T0* 
_class
loc:@conv4b/kernel*
validate_shape(*(
_output_shapes
:

conv4b/kernel/readIdentityconv4b/kernel*
T0* 
_class
loc:@conv4b/kernel*(
_output_shapes
:
[
conv4b/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
y
conv4b/bias
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ś
conv4b/bias/AssignAssignconv4b/biasconv4b/Const*
use_locking(*
T0*
_class
loc:@conv4b/bias*
validate_shape(*
_output_shapes	
:
o
conv4b/bias/readIdentityconv4b/bias*
T0*
_class
loc:@conv4b/bias*
_output_shapes	
:
q
 conv4b/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
č
conv4b/convolutionConv2Dconv4a/Reluconv4b/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

conv4b/BiasAddBiasAddconv4b/convolutionconv4b/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
`
conv4b/ReluReluconv4b/BiasAdd*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0
b
 drop4/keras_learning_phase/inputConst*
_output_shapes
: *
value	B
 Z *
dtype0


drop4/keras_learning_phasePlaceholderWithDefault drop4/keras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 
v
drop4/cond/SwitchSwitchdrop4/keras_learning_phasedrop4/keras_learning_phase*
T0
*
_output_shapes
: : 
U
drop4/cond/switch_tIdentitydrop4/cond/Switch:1*
_output_shapes
: *
T0

S
drop4/cond/switch_fIdentitydrop4/cond/Switch*
_output_shapes
: *
T0

[
drop4/cond/pred_idIdentitydrop4/keras_learning_phase*
T0
*
_output_shapes
: 
k
drop4/cond/mul/yConst^drop4/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
}
drop4/cond/mulMuldrop4/cond/mul/Switch:1drop4/cond/mul/y*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0
ť
drop4/cond/mul/SwitchSwitchconv4b/Reludrop4/cond/pred_id*
T0*
_class
loc:@conv4b/Relu*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙Ę:˙˙˙˙˙˙˙˙˙Ę
r
drop4/cond/dropout/rateConst^drop4/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
f
drop4/cond/dropout/ShapeShapedrop4/cond/mul*
T0*
out_type0*
_output_shapes
:
s
drop4/cond/dropout/sub/xConst^drop4/cond/switch_t*
_output_shapes
: *
valueB
 *  ?*
dtype0
q
drop4/cond/dropout/subSubdrop4/cond/dropout/sub/xdrop4/cond/dropout/rate*
T0*
_output_shapes
: 

%drop4/cond/dropout/random_uniform/minConst^drop4/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

%drop4/cond/dropout/random_uniform/maxConst^drop4/cond/switch_t*
_output_shapes
: *
valueB
 *  ?*
dtype0
Ă
/drop4/cond/dropout/random_uniform/RandomUniformRandomUniformdrop4/cond/dropout/Shape*
seed2ůă*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
seedą˙ĺ)*
T0*
dtype0

%drop4/cond/dropout/random_uniform/subSub%drop4/cond/dropout/random_uniform/max%drop4/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Á
%drop4/cond/dropout/random_uniform/mulMul/drop4/cond/dropout/random_uniform/RandomUniform%drop4/cond/dropout/random_uniform/sub*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
ł
!drop4/cond/dropout/random_uniformAdd%drop4/cond/dropout/random_uniform/mul%drop4/cond/dropout/random_uniform/min*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0

drop4/cond/dropout/addAdddrop4/cond/dropout/sub!drop4/cond/dropout/random_uniform*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
v
drop4/cond/dropout/FloorFloordrop4/cond/dropout/add*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

drop4/cond/dropout/truedivRealDivdrop4/cond/muldrop4/cond/dropout/sub*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

drop4/cond/dropout/mulMuldrop4/cond/dropout/truedivdrop4/cond/dropout/Floor*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
š
drop4/cond/Switch_1Switchconv4b/Reludrop4/cond/pred_id*
T0*
_class
loc:@conv4b/Relu*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙Ę:˙˙˙˙˙˙˙˙˙Ę

drop4/cond/MergeMergedrop4/cond/Switch_1drop4/cond/dropout/mul*
T0*
N*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙Ę: 
¸
pool4/MaxPoolMaxPooldrop4/cond/Merge*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
v
conv5a/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
a
conv5a/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv5a/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *Â<*
dtype0
¸
'conv5a/truncated_normal/TruncatedNormalTruncatedNormalconv5a/truncated_normal/shape*
seedą˙ĺ)*
T0*
dtype0*
seed2Šż*(
_output_shapes
:

conv5a/truncated_normal/mulMul'conv5a/truncated_normal/TruncatedNormalconv5a/truncated_normal/stddev*(
_output_shapes
:*
T0

conv5a/truncated_normalAddconv5a/truncated_normal/mulconv5a/truncated_normal/mean*
T0*(
_output_shapes
:

conv5a/kernel
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:*
shape:
Ä
conv5a/kernel/AssignAssignconv5a/kernelconv5a/truncated_normal*
T0* 
_class
loc:@conv5a/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(

conv5a/kernel/readIdentityconv5a/kernel*(
_output_shapes
:*
T0* 
_class
loc:@conv5a/kernel
[
conv5a/ConstConst*
_output_shapes	
:*
valueB*    *
dtype0
y
conv5a/bias
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ś
conv5a/bias/AssignAssignconv5a/biasconv5a/Const*
use_locking(*
T0*
_class
loc:@conv5a/bias*
validate_shape(*
_output_shapes	
:
o
conv5a/bias/readIdentityconv5a/bias*
_output_shapes	
:*
T0*
_class
loc:@conv5a/bias
q
 conv5a/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
č
conv5a/convolutionConv2Dpool4/MaxPoolconv5a/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
	dilations
*
T0

conv5a/BiasAddBiasAddconv5a/convolutionconv5a/bias/read*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
T0
^
conv5a/ReluReluconv5a/BiasAdd*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
v
conv5b/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
a
conv5b/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv5b/truncated_normal/stddevConst*
valueB
 *¸1<*
dtype0*
_output_shapes
: 
¸
'conv5b/truncated_normal/TruncatedNormalTruncatedNormalconv5b/truncated_normal/shape*
seedą˙ĺ)*
T0*
dtype0*
seed2üÍ*(
_output_shapes
:

conv5b/truncated_normal/mulMul'conv5b/truncated_normal/TruncatedNormalconv5b/truncated_normal/stddev*(
_output_shapes
:*
T0

conv5b/truncated_normalAddconv5b/truncated_normal/mulconv5b/truncated_normal/mean*
T0*(
_output_shapes
:

conv5b/kernel
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:*
shape:
Ä
conv5b/kernel/AssignAssignconv5b/kernelconv5b/truncated_normal*
use_locking(*
T0* 
_class
loc:@conv5b/kernel*
validate_shape(*(
_output_shapes
:

conv5b/kernel/readIdentityconv5b/kernel*
T0* 
_class
loc:@conv5b/kernel*(
_output_shapes
:
[
conv5b/ConstConst*
_output_shapes	
:*
valueB*    *
dtype0
y
conv5b/bias
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ś
conv5b/bias/AssignAssignconv5b/biasconv5b/Const*
_class
loc:@conv5b/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
o
conv5b/bias/readIdentityconv5b/bias*
_output_shapes	
:*
T0*
_class
loc:@conv5b/bias
q
 conv5b/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ć
conv5b/convolutionConv2Dconv5a/Reluconv5b/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

conv5b/BiasAddBiasAddconv5b/convolutionconv5b/bias/read*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
T0
^
conv5b/ReluReluconv5b/BiasAdd*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
v
drop5/cond/SwitchSwitchdrop4/keras_learning_phasedrop4/keras_learning_phase*
_output_shapes
: : *
T0

U
drop5/cond/switch_tIdentitydrop5/cond/Switch:1*
T0
*
_output_shapes
: 
S
drop5/cond/switch_fIdentitydrop5/cond/Switch*
T0
*
_output_shapes
: 
[
drop5/cond/pred_idIdentitydrop4/keras_learning_phase*
T0
*
_output_shapes
: 
k
drop5/cond/mul/yConst^drop5/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
{
drop5/cond/mulMuldrop5/cond/mul/Switch:1drop5/cond/mul/y*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
ˇ
drop5/cond/mul/SwitchSwitchconv5b/Reludrop5/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙Ce:˙˙˙˙˙˙˙˙˙Ce*
T0*
_class
loc:@conv5b/Relu
r
drop5/cond/dropout/rateConst^drop5/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
f
drop5/cond/dropout/ShapeShapedrop5/cond/mul*
T0*
out_type0*
_output_shapes
:
s
drop5/cond/dropout/sub/xConst^drop5/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
q
drop5/cond/dropout/subSubdrop5/cond/dropout/sub/xdrop5/cond/dropout/rate*
_output_shapes
: *
T0

%drop5/cond/dropout/random_uniform/minConst^drop5/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

%drop5/cond/dropout/random_uniform/maxConst^drop5/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Á
/drop5/cond/dropout/random_uniform/RandomUniformRandomUniformdrop5/cond/dropout/Shape*
seedą˙ĺ)*
T0*
dtype0*
seed2ŁĎ*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

%drop5/cond/dropout/random_uniform/subSub%drop5/cond/dropout/random_uniform/max%drop5/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
ż
%drop5/cond/dropout/random_uniform/mulMul/drop5/cond/dropout/random_uniform/RandomUniform%drop5/cond/dropout/random_uniform/sub*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
T0
ą
!drop5/cond/dropout/random_uniformAdd%drop5/cond/dropout/random_uniform/mul%drop5/cond/dropout/random_uniform/min*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

drop5/cond/dropout/addAdddrop5/cond/dropout/sub!drop5/cond/dropout/random_uniform*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
t
drop5/cond/dropout/FloorFloordrop5/cond/dropout/add*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
T0

drop5/cond/dropout/truedivRealDivdrop5/cond/muldrop5/cond/dropout/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

drop5/cond/dropout/mulMuldrop5/cond/dropout/truedivdrop5/cond/dropout/Floor*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
ľ
drop5/cond/Switch_1Switchconv5b/Reludrop5/cond/pred_id*
_class
loc:@conv5b/Relu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙Ce:˙˙˙˙˙˙˙˙˙Ce*
T0

drop5/cond/MergeMergedrop5/cond/Switch_1drop5/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ce: *
T0*
N
Y
	up1/ShapeShapedrop5/cond/Merge*
out_type0*
_output_shapes
:*
T0
a
up1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
c
up1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
c
up1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

up1/strided_sliceStridedSlice	up1/Shapeup1/strided_slice/stackup1/strided_slice/stack_1up1/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask 
Z
	up1/ConstConst*
dtype0*
_output_shapes
:*
valueB"      
Q
up1/mulMulup1/strided_slice	up1/Const*
T0*
_output_shapes
:

up1/ResizeNearestNeighborResizeNearestNeighbordrop5/cond/Mergeup1/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
align_corners( *
T0
u
conv6/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
`
conv6/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
b
conv6/truncated_normal/stddevConst*
valueB
 *ĘÍ<*
dtype0*
_output_shapes
: 
ś
&conv6/truncated_normal/TruncatedNormalTruncatedNormalconv6/truncated_normal/shape*
seed2Łż*(
_output_shapes
:*
seedą˙ĺ)*
T0*
dtype0

conv6/truncated_normal/mulMul&conv6/truncated_normal/TruncatedNormalconv6/truncated_normal/stddev*(
_output_shapes
:*
T0

conv6/truncated_normalAddconv6/truncated_normal/mulconv6/truncated_normal/mean*
T0*(
_output_shapes
:

conv6/kernel
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
Ŕ
conv6/kernel/AssignAssignconv6/kernelconv6/truncated_normal*(
_output_shapes
:*
use_locking(*
T0*
_class
loc:@conv6/kernel*
validate_shape(

conv6/kernel/readIdentityconv6/kernel*
T0*
_class
loc:@conv6/kernel*(
_output_shapes
:
Z
conv6/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
x

conv6/bias
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
˘
conv6/bias/AssignAssign
conv6/biasconv6/Const*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv6/bias*
validate_shape(
l
conv6/bias/readIdentity
conv6/bias*
T0*
_class
loc:@conv6/bias*
_output_shapes	
:
p
conv6/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ô
conv6/convolutionConv2Dup1/ResizeNearestNeighborconv6/kernel/read*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

conv6/BiasAddBiasAddconv6/convolutionconv6/bias/read*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0
^

conv6/ReluReluconv6/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*9
value0B."                                *
dtype0

zero_padding2d_1/PadPad
conv6/Reluzero_padding2d_1/Pad/paddings*
T0*
	Tpaddings0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
[
concatenate_1/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ľ
concatenate_1/concatConcatV2drop4/cond/Mergezero_padding2d_1/Padconcatenate_1/concat/axis*

Tidx0*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
v
conv7a/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
a
conv7a/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv7a/truncated_normal/stddevConst*
valueB
 *¸1<*
dtype0*
_output_shapes
: 
¸
'conv7a/truncated_normal/TruncatedNormalTruncatedNormalconv7a/truncated_normal/shape*
T0*
dtype0*
seed2Đůç*(
_output_shapes
:*
seedą˙ĺ)

conv7a/truncated_normal/mulMul'conv7a/truncated_normal/TruncatedNormalconv7a/truncated_normal/stddev*(
_output_shapes
:*
T0

conv7a/truncated_normalAddconv7a/truncated_normal/mulconv7a/truncated_normal/mean*
T0*(
_output_shapes
:

conv7a/kernel
VariableV2*
shape:*
shared_name *
dtype0*
	container *(
_output_shapes
:
Ä
conv7a/kernel/AssignAssignconv7a/kernelconv7a/truncated_normal*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv7a/kernel

conv7a/kernel/readIdentityconv7a/kernel* 
_class
loc:@conv7a/kernel*(
_output_shapes
:*
T0
[
conv7a/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
y
conv7a/bias
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ś
conv7a/bias/AssignAssignconv7a/biasconv7a/Const*
use_locking(*
T0*
_class
loc:@conv7a/bias*
validate_shape(*
_output_shapes	
:
o
conv7a/bias/readIdentityconv7a/bias*
T0*
_class
loc:@conv7a/bias*
_output_shapes	
:
q
 conv7a/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
ń
conv7a/convolutionConv2Dconcatenate_1/concatconv7a/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations


conv7a/BiasAddBiasAddconv7a/convolutionconv7a/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
`
conv7a/ReluReluconv7a/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
v
conv7b/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
a
conv7b/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
conv7b/truncated_normal/stddevConst*
valueB
 *Â<*
dtype0*
_output_shapes
: 
¸
'conv7b/truncated_normal/TruncatedNormalTruncatedNormalconv7b/truncated_normal/shape*
dtype0*
seed2ÍĄţ*(
_output_shapes
:*
seedą˙ĺ)*
T0

conv7b/truncated_normal/mulMul'conv7b/truncated_normal/TruncatedNormalconv7b/truncated_normal/stddev*(
_output_shapes
:*
T0

conv7b/truncated_normalAddconv7b/truncated_normal/mulconv7b/truncated_normal/mean*
T0*(
_output_shapes
:

conv7b/kernel
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:*
shape:
Ä
conv7b/kernel/AssignAssignconv7b/kernelconv7b/truncated_normal*
T0* 
_class
loc:@conv7b/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(

conv7b/kernel/readIdentityconv7b/kernel*
T0* 
_class
loc:@conv7b/kernel*(
_output_shapes
:
[
conv7b/ConstConst*
_output_shapes	
:*
valueB*    *
dtype0
y
conv7b/bias
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ś
conv7b/bias/AssignAssignconv7b/biasconv7b/Const*
use_locking(*
T0*
_class
loc:@conv7b/bias*
validate_shape(*
_output_shapes	
:
o
conv7b/bias/readIdentityconv7b/bias*
_class
loc:@conv7b/bias*
_output_shapes	
:*
T0
q
 conv7b/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
č
conv7b/convolutionConv2Dconv7a/Reluconv7b/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

conv7b/BiasAddBiasAddconv7b/convolutionconv7b/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
`
conv7b/ReluReluconv7b/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
`
up_sampling2d_1/ShapeShapeconv7b/Relu*
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
%up_sampling2d_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Í
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape#up_sampling2d_1/strided_slice/stack%up_sampling2d_1/strided_slice/stack_1%up_sampling2d_1/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
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
˛
%up_sampling2d_1/ResizeNearestNeighborResizeNearestNeighborconv7b/Reluup_sampling2d_1/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
align_corners( *
T0
x
conv2d_1/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
c
conv2d_1/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
e
 conv2d_1/truncated_normal/stddevConst*
valueB
 *6=*
dtype0*
_output_shapes
: 
ź
)conv2d_1/truncated_normal/TruncatedNormalTruncatedNormalconv2d_1/truncated_normal/shape*
seedą˙ĺ)*
T0*
dtype0*
seed2ŐĽ*(
_output_shapes
:
¤
conv2d_1/truncated_normal/mulMul)conv2d_1/truncated_normal/TruncatedNormal conv2d_1/truncated_normal/stddev*(
_output_shapes
:*
T0

conv2d_1/truncated_normalAddconv2d_1/truncated_normal/mulconv2d_1/truncated_normal/mean*
T0*(
_output_shapes
:

conv2d_1/kernel
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
Ě
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/truncated_normal*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel

conv2d_1/kernel/readIdentityconv2d_1/kernel*(
_output_shapes
:*
T0*"
_class
loc:@conv2d_1/kernel
]
conv2d_1/ConstConst*
_output_shapes	
:*
valueB*    *
dtype0
{
conv2d_1/bias
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ž
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
u
conv2d_1/bias/readIdentityconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
_output_shapes	
:*
T0
s
"conv2d_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      

conv2d_1/convolutionConv2D%up_sampling2d_1/ResizeNearestNeighborconv2d_1/kernel/read*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC

conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0
d
conv2d_1/ReluReluconv2d_1/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
[
concatenate_2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Š
concatenate_2/concatConcatV2conv3b/Reluconv2d_1/Reluconcatenate_2/concat/axis*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0
x
conv2d_2/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
c
conv2d_2/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 conv2d_2/truncated_normal/stddevConst*
valueB
 *Â<*
dtype0*
_output_shapes
: 
ź
)conv2d_2/truncated_normal/TruncatedNormalTruncatedNormalconv2d_2/truncated_normal/shape*
T0*
dtype0*
seed2ĄÍĺ*(
_output_shapes
:*
seedą˙ĺ)
¤
conv2d_2/truncated_normal/mulMul)conv2d_2/truncated_normal/TruncatedNormal conv2d_2/truncated_normal/stddev*
T0*(
_output_shapes
:

conv2d_2/truncated_normalAddconv2d_2/truncated_normal/mulconv2d_2/truncated_normal/mean*
T0*(
_output_shapes
:

conv2d_2/kernel
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
Ě
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/truncated_normal*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel

conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*(
_output_shapes
:
]
conv2d_2/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_2/bias
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ž
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes	
:
u
conv2d_2/bias/readIdentityconv2d_2/bias*
_output_shapes	
:*
T0* 
_class
loc:@conv2d_2/bias
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ő
conv2d_2/convolutionConv2Dconcatenate_2/concatconv2d_2/kernel/read*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
d
conv2d_2/ReluReluconv2d_2/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
x
conv2d_3/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
c
conv2d_3/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 conv2d_3/truncated_normal/stddevConst*
valueB
 *¸1	=*
dtype0*
_output_shapes
: 
ź
)conv2d_3/truncated_normal/TruncatedNormalTruncatedNormalconv2d_3/truncated_normal/shape*
T0*
dtype0*
seed2ţŞĂ*(
_output_shapes
:*
seedą˙ĺ)
¤
conv2d_3/truncated_normal/mulMul)conv2d_3/truncated_normal/TruncatedNormal conv2d_3/truncated_normal/stddev*(
_output_shapes
:*
T0

conv2d_3/truncated_normalAddconv2d_3/truncated_normal/mulconv2d_3/truncated_normal/mean*(
_output_shapes
:*
T0

conv2d_3/kernel
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
Ě
conv2d_3/kernel/AssignAssignconv2d_3/kernelconv2d_3/truncated_normal*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(*(
_output_shapes
:

conv2d_3/kernel/readIdentityconv2d_3/kernel*
T0*"
_class
loc:@conv2d_3/kernel*(
_output_shapes
:
]
conv2d_3/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_3/bias
VariableV2*
	container *
_output_shapes	
:*
shape:*
shared_name *
dtype0
Ž
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/Const*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(
u
conv2d_3/bias/readIdentityconv2d_3/bias*
T0* 
_class
loc:@conv2d_3/bias*
_output_shapes	
:
s
"conv2d_3/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
î
conv2d_3/convolutionConv2Dconv2d_2/Reluconv2d_3/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙

conv2d_3/BiasAddBiasAddconv2d_3/convolutionconv2d_3/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
d
conv2d_3/ReluReluconv2d_3/BiasAdd*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0
b
up_sampling2d_2/ShapeShapeconv2d_3/Relu*
out_type0*
_output_shapes
:*
T0
m
#up_sampling2d_2/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
o
%up_sampling2d_2/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
o
%up_sampling2d_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Í
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape#up_sampling2d_2/strided_slice/stack%up_sampling2d_2/strided_slice/stack_1%up_sampling2d_2/strided_slice/stack_2*
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
up_sampling2d_2/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
u
up_sampling2d_2/mulMulup_sampling2d_2/strided_sliceup_sampling2d_2/Const*
T0*
_output_shapes
:
´
%up_sampling2d_2/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Reluup_sampling2d_2/mul*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
align_corners( 
x
conv2d_4/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
c
conv2d_4/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 conv2d_4/truncated_normal/stddevConst*
valueB
 *ĘM=*
dtype0*
_output_shapes
: 
ź
)conv2d_4/truncated_normal/TruncatedNormalTruncatedNormalconv2d_4/truncated_normal/shape*
dtype0*
seed2áé*(
_output_shapes
:*
seedą˙ĺ)*
T0
¤
conv2d_4/truncated_normal/mulMul)conv2d_4/truncated_normal/TruncatedNormal conv2d_4/truncated_normal/stddev*(
_output_shapes
:*
T0

conv2d_4/truncated_normalAddconv2d_4/truncated_normal/mulconv2d_4/truncated_normal/mean*(
_output_shapes
:*
T0

conv2d_4/kernel
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
Ě
conv2d_4/kernel/AssignAssignconv2d_4/kernelconv2d_4/truncated_normal*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*(
_output_shapes
:

conv2d_4/kernel/readIdentityconv2d_4/kernel*
T0*"
_class
loc:@conv2d_4/kernel*(
_output_shapes
:
]
conv2d_4/ConstConst*
_output_shapes	
:*
valueB*    *
dtype0
{
conv2d_4/bias
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
Ž
conv2d_4/bias/AssignAssignconv2d_4/biasconv2d_4/Const*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(
u
conv2d_4/bias/readIdentityconv2d_4/bias*
_output_shapes	
:*
T0* 
_class
loc:@conv2d_4/bias
s
"conv2d_4/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

conv2d_4/convolutionConv2D%up_sampling2d_2/ResizeNearestNeighborconv2d_4/kernel/read*2
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
conv2d_4/BiasAddBiasAddconv2d_4/convolutionconv2d_4/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
d
conv2d_4/ReluReluconv2d_4/BiasAdd*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
[
concatenate_3/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Š
concatenate_3/concatConcatV2conv2b/Reluconv2d_4/Reluconcatenate_3/concat/axis*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*

Tidx0*
T0
x
conv2d_5/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
c
conv2d_5/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 conv2d_5/truncated_normal/stddevConst*
valueB
 *¸1	=*
dtype0*
_output_shapes
: 
ť
)conv2d_5/truncated_normal/TruncatedNormalTruncatedNormalconv2d_5/truncated_normal/shape*
dtype0*
seed2â¨y*(
_output_shapes
:*
seedą˙ĺ)*
T0
¤
conv2d_5/truncated_normal/mulMul)conv2d_5/truncated_normal/TruncatedNormal conv2d_5/truncated_normal/stddev*
T0*(
_output_shapes
:

conv2d_5/truncated_normalAddconv2d_5/truncated_normal/mulconv2d_5/truncated_normal/mean*
T0*(
_output_shapes
:

conv2d_5/kernel
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
Ě
conv2d_5/kernel/AssignAssignconv2d_5/kernelconv2d_5/truncated_normal*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel

conv2d_5/kernel/readIdentityconv2d_5/kernel*
T0*"
_class
loc:@conv2d_5/kernel*(
_output_shapes
:
]
conv2d_5/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_5/bias
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ž
conv2d_5/bias/AssignAssignconv2d_5/biasconv2d_5/Const*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias*
validate_shape(*
_output_shapes	
:
u
conv2d_5/bias/readIdentityconv2d_5/bias*
T0* 
_class
loc:@conv2d_5/bias*
_output_shapes	
:
s
"conv2d_5/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ő
conv2d_5/convolutionConv2Dconcatenate_3/concatconv2d_5/kernel/read*
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
conv2d_5/BiasAddBiasAddconv2d_5/convolutionconv2d_5/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
d
conv2d_5/ReluReluconv2d_5/BiasAdd*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0
x
conv2d_6/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
c
conv2d_6/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 conv2d_6/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *B=
ź
)conv2d_6/truncated_normal/TruncatedNormalTruncatedNormalconv2d_6/truncated_normal/shape*
T0*
dtype0*
seed2ŤŐ*(
_output_shapes
:*
seedą˙ĺ)
¤
conv2d_6/truncated_normal/mulMul)conv2d_6/truncated_normal/TruncatedNormal conv2d_6/truncated_normal/stddev*
T0*(
_output_shapes
:

conv2d_6/truncated_normalAddconv2d_6/truncated_normal/mulconv2d_6/truncated_normal/mean*
T0*(
_output_shapes
:

conv2d_6/kernel
VariableV2*
	container *(
_output_shapes
:*
shape:*
shared_name *
dtype0
Ě
conv2d_6/kernel/AssignAssignconv2d_6/kernelconv2d_6/truncated_normal*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*(
_output_shapes
:

conv2d_6/kernel/readIdentityconv2d_6/kernel*
T0*"
_class
loc:@conv2d_6/kernel*(
_output_shapes
:
]
conv2d_6/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
{
conv2d_6/bias
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ž
conv2d_6/bias/AssignAssignconv2d_6/biasconv2d_6/Const*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@conv2d_6/bias
u
conv2d_6/bias/readIdentityconv2d_6/bias*
T0* 
_class
loc:@conv2d_6/bias*
_output_shapes	
:
s
"conv2d_6/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
î
conv2d_6/convolutionConv2Dconv2d_5/Reluconv2d_6/kernel/read*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv2d_6/BiasAddBiasAddconv2d_6/convolutionconv2d_6/bias/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
d
conv2d_6/ReluReluconv2d_6/BiasAdd*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0
b
up_sampling2d_3/ShapeShapeconv2d_6/Relu*
_output_shapes
:*
T0*
out_type0
m
#up_sampling2d_3/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
o
%up_sampling2d_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Í
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape#up_sampling2d_3/strided_slice/stack%up_sampling2d_3/strided_slice/stack_1%up_sampling2d_3/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:
f
up_sampling2d_3/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
u
up_sampling2d_3/mulMulup_sampling2d_3/strided_sliceup_sampling2d_3/Const*
_output_shapes
:*
T0
´
%up_sampling2d_3/ResizeNearestNeighborResizeNearestNeighborconv2d_6/Reluup_sampling2d_3/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¸Đ*
align_corners( *
T0
x
conv2d_7/truncated_normal/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
c
conv2d_7/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 conv2d_7/truncated_normal/stddevConst*
valueB
 *6=*
dtype0*
_output_shapes
: 
ť
)conv2d_7/truncated_normal/TruncatedNormalTruncatedNormalconv2d_7/truncated_normal/shape*
T0*
dtype0*
seed2÷áĹ*'
_output_shapes
:@*
seedą˙ĺ)
Ł
conv2d_7/truncated_normal/mulMul)conv2d_7/truncated_normal/TruncatedNormal conv2d_7/truncated_normal/stddev*'
_output_shapes
:@*
T0

conv2d_7/truncated_normalAddconv2d_7/truncated_normal/mulconv2d_7/truncated_normal/mean*
T0*'
_output_shapes
:@

conv2d_7/kernel
VariableV2*
dtype0*
	container *'
_output_shapes
:@*
shape:@*
shared_name 
Ë
conv2d_7/kernel/AssignAssignconv2d_7/kernelconv2d_7/truncated_normal*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@conv2d_7/kernel

conv2d_7/kernel/readIdentityconv2d_7/kernel*
T0*"
_class
loc:@conv2d_7/kernel*'
_output_shapes
:@
[
conv2d_7/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_7/bias
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
­
conv2d_7/bias/AssignAssignconv2d_7/biasconv2d_7/Const*
T0* 
_class
loc:@conv2d_7/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
t
conv2d_7/bias/readIdentityconv2d_7/bias* 
_class
loc:@conv2d_7/bias*
_output_shapes
:@*
T0
s
"conv2d_7/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

conv2d_7/convolutionConv2D%up_sampling2d_3/ResizeNearestNeighborconv2d_7/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@

conv2d_7/BiasAddBiasAddconv2d_7/convolutionconv2d_7/bias/read*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0
c
conv2d_7/ReluReluconv2d_7/BiasAdd*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0
[
concatenate_4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Š
concatenate_4/concatConcatV2conv1b/Reluconv2d_7/Reluconcatenate_4/concat/axis*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¸Đ*

Tidx0
x
conv2d_8/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"         @   *
dtype0
c
conv2d_8/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 conv2d_8/truncated_normal/stddevConst*
valueB
 *B=*
dtype0*
_output_shapes
: 
ť
)conv2d_8/truncated_normal/TruncatedNormalTruncatedNormalconv2d_8/truncated_normal/shape*
dtype0*
seed2ËËĽ*'
_output_shapes
:@*
seedą˙ĺ)*
T0
Ł
conv2d_8/truncated_normal/mulMul)conv2d_8/truncated_normal/TruncatedNormal conv2d_8/truncated_normal/stddev*'
_output_shapes
:@*
T0

conv2d_8/truncated_normalAddconv2d_8/truncated_normal/mulconv2d_8/truncated_normal/mean*'
_output_shapes
:@*
T0

conv2d_8/kernel
VariableV2*
dtype0*
	container *'
_output_shapes
:@*
shape:@*
shared_name 
Ë
conv2d_8/kernel/AssignAssignconv2d_8/kernelconv2d_8/truncated_normal*'
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@conv2d_8/kernel*
validate_shape(

conv2d_8/kernel/readIdentityconv2d_8/kernel*
T0*"
_class
loc:@conv2d_8/kernel*'
_output_shapes
:@
[
conv2d_8/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_8/bias
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
­
conv2d_8/bias/AssignAssignconv2d_8/biasconv2d_8/Const*
T0* 
_class
loc:@conv2d_8/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
t
conv2d_8/bias/readIdentityconv2d_8/bias*
T0* 
_class
loc:@conv2d_8/bias*
_output_shapes
:@
s
"conv2d_8/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
ô
conv2d_8/convolutionConv2Dconcatenate_4/concatconv2d_8/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
	dilations


conv2d_8/BiasAddBiasAddconv2d_8/convolutionconv2d_8/bias/read*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
T0
c
conv2d_8/ReluReluconv2d_8/BiasAdd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
x
conv2d_9/truncated_normal/shapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
c
conv2d_9/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
 conv2d_9/truncated_normal/stddevConst*
valueB
 *¸1=*
dtype0*
_output_shapes
: 
š
)conv2d_9/truncated_normal/TruncatedNormalTruncatedNormalconv2d_9/truncated_normal/shape*
T0*
dtype0*
seed2Óť*&
_output_shapes
:@@*
seedą˙ĺ)
˘
conv2d_9/truncated_normal/mulMul)conv2d_9/truncated_normal/TruncatedNormal conv2d_9/truncated_normal/stddev*&
_output_shapes
:@@*
T0

conv2d_9/truncated_normalAddconv2d_9/truncated_normal/mulconv2d_9/truncated_normal/mean*
T0*&
_output_shapes
:@@

conv2d_9/kernel
VariableV2*
dtype0*
	container *&
_output_shapes
:@@*
shape:@@*
shared_name 
Ę
conv2d_9/kernel/AssignAssignconv2d_9/kernelconv2d_9/truncated_normal*&
_output_shapes
:@@*
use_locking(*
T0*"
_class
loc:@conv2d_9/kernel*
validate_shape(

conv2d_9/kernel/readIdentityconv2d_9/kernel*
T0*"
_class
loc:@conv2d_9/kernel*&
_output_shapes
:@@
[
conv2d_9/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_9/bias
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
­
conv2d_9/bias/AssignAssignconv2d_9/biasconv2d_9/Const*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@conv2d_9/bias
t
conv2d_9/bias/readIdentityconv2d_9/bias*
T0* 
_class
loc:@conv2d_9/bias*
_output_shapes
:@
s
"conv2d_9/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
í
conv2d_9/convolutionConv2Dconv2d_8/Reluconv2d_9/kernel/read*1
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
conv2d_9/BiasAddBiasAddconv2d_9/convolutionconv2d_9/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
c
conv2d_9/ReluReluconv2d_9/BiasAdd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
w
conv2d_10/random_uniform/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
a
conv2d_10/random_uniform/minConst*
valueB
 *Ş7ž*
dtype0*
_output_shapes
: 
a
conv2d_10/random_uniform/maxConst*
valueB
 *Ş7>*
dtype0*
_output_shapes
: 
ł
&conv2d_10/random_uniform/RandomUniformRandomUniformconv2d_10/random_uniform/shape*
seedą˙ĺ)*
T0*
dtype0*
seed2É˝*&
_output_shapes
:@

conv2d_10/random_uniform/subSubconv2d_10/random_uniform/maxconv2d_10/random_uniform/min*
_output_shapes
: *
T0

conv2d_10/random_uniform/mulMul&conv2d_10/random_uniform/RandomUniformconv2d_10/random_uniform/sub*&
_output_shapes
:@*
T0

conv2d_10/random_uniformAddconv2d_10/random_uniform/mulconv2d_10/random_uniform/min*&
_output_shapes
:@*
T0

conv2d_10/kernel
VariableV2*
shared_name *
dtype0*
	container *&
_output_shapes
:@*
shape:@
Ě
conv2d_10/kernel/AssignAssignconv2d_10/kernelconv2d_10/random_uniform*
use_locking(*
T0*#
_class
loc:@conv2d_10/kernel*
validate_shape(*&
_output_shapes
:@

conv2d_10/kernel/readIdentityconv2d_10/kernel*
T0*#
_class
loc:@conv2d_10/kernel*&
_output_shapes
:@
\
conv2d_10/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
z
conv2d_10/bias
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
ą
conv2d_10/bias/AssignAssignconv2d_10/biasconv2d_10/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@conv2d_10/bias
w
conv2d_10/bias/readIdentityconv2d_10/bias*
T0*!
_class
loc:@conv2d_10/bias*
_output_shapes
:
t
#conv2d_10/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
đ
conv2d_10/convolutionConv2Dconv2d_9/Reluconv2d_10/kernel/read*
use_cudnn_on_gpu(*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
	dilations
*
T0*
data_formatNHWC*
strides


conv2d_10/BiasAddBiasAddconv2d_10/convolutionconv2d_10/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
k
conv2d_10/SigmoidSigmoidconv2d_10/BiasAdd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
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
 *ˇŃ8*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
shape: 
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
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
Ž
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
Adam/beta_2/readIdentityAdam/beta_2*
T0*
_class
loc:@Adam/beta_2*
_output_shapes
: 
]
Adam/decay/initial_valueConst*
valueB
 *ŹĹ'7*
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
Adam/decay*
T0*
_class
loc:@Adam/decay*
_output_shapes
: 
š
conv2d_10_targetPlaceholder*
dtype0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*?
shape6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
s
conv2d_10_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙

loss/conv2d_10_loss/subSubconv2d_10/Sigmoidconv2d_10_target*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
T0
s
loss/conv2d_10_loss/AbsAbsloss/conv2d_10_loss/sub*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
T0
u
*loss/conv2d_10_loss/Mean/reduction_indicesConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
ş
loss/conv2d_10_loss/MeanMeanloss/conv2d_10_loss/Abs*loss/conv2d_10_loss/Mean/reduction_indices*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*

Tidx0*
	keep_dims( 
}
,loss/conv2d_10_loss/Mean_1/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
ľ
loss/conv2d_10_loss/Mean_1Meanloss/conv2d_10_loss/Mean,loss/conv2d_10_loss/Mean_1/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
	keep_dims( *
T0

loss/conv2d_10_loss/mulMulloss/conv2d_10_loss/Mean_1conv2d_10_sample_weights*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
loss/conv2d_10_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/conv2d_10_loss/NotEqualNotEqualconv2d_10_sample_weightsloss/conv2d_10_loss/NotEqual/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

loss/conv2d_10_loss/CastCastloss/conv2d_10_loss/NotEqual*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
loss/conv2d_10_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/conv2d_10_loss/Mean_2Meanloss/conv2d_10_loss/Castloss/conv2d_10_loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0

loss/conv2d_10_loss/truedivRealDivloss/conv2d_10_loss/mulloss/conv2d_10_loss/Mean_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
loss/conv2d_10_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0

loss/conv2d_10_loss/Mean_3Meanloss/conv2d_10_loss/truedivloss/conv2d_10_loss/Const_1*
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
X
loss/mulMul
loss/mul/xloss/conv2d_10_loss/Mean_3*
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
¨
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/conv2d_10_loss/Mean_3*
T0*
_class
loc:@loss/mul*
_output_shapes
: 

+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
ž
Etraining/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Reshape/shapeConst*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
valueB:*
dtype0*
_output_shapes
:
 
?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Etraining/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Reshape/shape*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
Tshape0*
_output_shapes
:
Ç
=training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/ShapeShapeloss/conv2d_10_loss/truediv*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
out_type0*
_output_shapes
:
ł
<training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/TileTile?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Reshape=training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3
É
?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Shape_1Shapeloss/conv2d_10_loss/truediv*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
out_type0*
_output_shapes
:*
T0
ą
?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Shape_2Const*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
valueB *
dtype0*
_output_shapes
: 
ś
=training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/ConstConst*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
valueB: *
dtype0*
_output_shapes
:
ą
<training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/ProdProd?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Shape_1=training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3
¸
?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Const_1Const*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
valueB: *
dtype0*
_output_shapes
:
ľ
>training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Prod_1Prod?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Shape_2?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Const_1*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
_output_shapes
: 
˛
Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Maximum/yConst*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
value	B :*
dtype0*
_output_shapes
: 

?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/MaximumMaximum>training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Prod_1Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Maximum/y*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
_output_shapes
: 

@training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/floordivFloorDiv<training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Prod?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Maximum*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
_output_shapes
: 
ő
<training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/CastCast@training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/floordiv*

SrcT0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*
Truncate( *

DstT0*
_output_shapes
: 
Ł
?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/truedivRealDiv<training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Tile<training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/Cast*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
>training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/ShapeShapeloss/conv2d_10_loss/mul*
T0*.
_class$
" loc:@loss/conv2d_10_loss/truediv*
out_type0*
_output_shapes
:
ł
@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Shape_1Const*.
_class$
" loc:@loss/conv2d_10_loss/truediv*
valueB *
dtype0*
_output_shapes
: 
Ö
Ntraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs>training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Shape@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Shape_1*.
_class$
" loc:@loss/conv2d_10_loss/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/RealDivRealDiv?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/truedivloss/conv2d_10_loss/Mean_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*.
_class$
" loc:@loss/conv2d_10_loss/truediv
Ĺ
<training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/SumSum@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/RealDivNtraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@loss/conv2d_10_loss/truediv*
_output_shapes
:
ľ
@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/ReshapeReshape<training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Sum>training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Shape*
T0*.
_class$
" loc:@loss/conv2d_10_loss/truediv*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
<training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/NegNegloss/conv2d_10_loss/mul*
T0*.
_class$
" loc:@loss/conv2d_10_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Btraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/RealDiv_1RealDiv<training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Negloss/conv2d_10_loss/Mean_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*.
_class$
" loc:@loss/conv2d_10_loss/truediv

Btraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/RealDiv_2RealDivBtraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/RealDiv_1loss/conv2d_10_loss/Mean_2*.
_class$
" loc:@loss/conv2d_10_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
<training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/mulMul?training/Adam/gradients/loss/conv2d_10_loss/Mean_3_grad/truedivBtraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/RealDiv_2*
T0*.
_class$
" loc:@loss/conv2d_10_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
>training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Sum_1Sum<training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/mulPtraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@loss/conv2d_10_loss/truediv
Ž
Btraining/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Reshape_1Reshape>training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Sum_1@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Shape_1*
T0*.
_class$
" loc:@loss/conv2d_10_loss/truediv*
Tshape0*
_output_shapes
: 
Ŕ
:training/Adam/gradients/loss/conv2d_10_loss/mul_grad/ShapeShapeloss/conv2d_10_loss/Mean_1*
T0**
_class 
loc:@loss/conv2d_10_loss/mul*
out_type0*
_output_shapes
:
Ŕ
<training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Shape_1Shapeconv2d_10_sample_weights*
_output_shapes
:*
T0**
_class 
loc:@loss/conv2d_10_loss/mul*
out_type0
Ć
Jtraining/Adam/gradients/loss/conv2d_10_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Shape<training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Shape_1*
T0**
_class 
loc:@loss/conv2d_10_loss/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ő
8training/Adam/gradients/loss/conv2d_10_loss/mul_grad/MulMul@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Reshapeconv2d_10_sample_weights*
T0**
_class 
loc:@loss/conv2d_10_loss/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
8training/Adam/gradients/loss/conv2d_10_loss/mul_grad/SumSum8training/Adam/gradients/loss/conv2d_10_loss/mul_grad/MulJtraining/Adam/gradients/loss/conv2d_10_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0**
_class 
loc:@loss/conv2d_10_loss/mul
Ľ
<training/Adam/gradients/loss/conv2d_10_loss/mul_grad/ReshapeReshape8training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Sum:training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Shape*
T0**
_class 
loc:@loss/conv2d_10_loss/mul*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
:training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Mul_1Mulloss/conv2d_10_loss/Mean_1@training/Adam/gradients/loss/conv2d_10_loss/truediv_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0**
_class 
loc:@loss/conv2d_10_loss/mul
ˇ
:training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Sum_1Sum:training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Mul_1Ltraining/Adam/gradients/loss/conv2d_10_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0**
_class 
loc:@loss/conv2d_10_loss/mul*
_output_shapes
:
Ť
>training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Reshape_1Reshape:training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Sum_1<training/Adam/gradients/loss/conv2d_10_loss/mul_grad/Shape_1**
_class 
loc:@loss/conv2d_10_loss/mul*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ä
=training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/ShapeShapeloss/conv2d_10_loss/Mean*
_output_shapes
:*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
out_type0
­
<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/SizeConst*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 

;training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/addAdd,loss/conv2d_10_loss/Mean_1/reduction_indices<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Size*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
_output_shapes
:

;training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/modFloorMod;training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/add<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Size*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
_output_shapes
:
¸
?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape_1Const*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
valueB:*
dtype0*
_output_shapes
:
´
Ctraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/range/startConst*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
value	B : *
dtype0*
_output_shapes
: 
´
Ctraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/range/deltaConst*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
ĺ
=training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/rangeRangeCtraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/range/start<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/SizeCtraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/range/delta*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
_output_shapes
:*

Tidx0
ł
Btraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Fill/valueConst*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
Ż
<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/FillFill?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape_1Btraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Fill/value*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*

index_type0*
_output_shapes
:*
T0
Ź
Etraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/DynamicStitchDynamicStitch=training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/range;training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/mod=training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Fill*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
N*
_output_shapes
:
˛
Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Maximum/yConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
value	B :
¨
?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/MaximumMaximumEtraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/DynamicStitchAtraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Maximum/y*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
_output_shapes
:
 
@training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/floordivFloorDiv=training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Maximum*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
_output_shapes
:*
T0
Ô
?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/ReshapeReshape<training/Adam/gradients/loss/conv2d_10_loss/mul_grad/ReshapeEtraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/DynamicStitch*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
Tshape0*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
Đ
<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/TileTile?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Reshape@training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/floordiv*

Tmultiples0*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ć
?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape_2Shapeloss/conv2d_10_loss/Mean*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
out_type0*
_output_shapes
:*
T0
Č
?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape_3Shapeloss/conv2d_10_loss/Mean_1*
_output_shapes
:*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
out_type0
ś
=training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/ConstConst*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
ą
<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/ProdProd?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape_2=training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Const*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
_output_shapes
: 
¸
?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Const_1Const*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
ľ
>training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Prod_1Prod?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Shape_3?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1
´
Ctraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Maximum_1/yConst*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
Ą
Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Maximum_1Maximum>training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Prod_1Ctraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Maximum_1/y*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
_output_shapes
: 

Btraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/floordiv_1FloorDiv<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/ProdAtraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Maximum_1*
T0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
_output_shapes
: 
÷
<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/CastCastBtraining/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/floordiv_1*

DstT0*
_output_shapes
: *

SrcT0*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*
Truncate( 
­
?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/truedivRealDiv<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Tile<training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/Cast*-
_class#
!loc:@loss/conv2d_10_loss/Mean_1*-
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
T0
ż
;training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/ShapeShapeloss/conv2d_10_loss/Abs*
_output_shapes
:*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
out_type0
Š
:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/SizeConst*+
_class!
loc:@loss/conv2d_10_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
ö
9training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/addAdd*loss/conv2d_10_loss/Mean/reduction_indices:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Size*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
: 

9training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/modFloorMod9training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/add:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Size*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
: 
­
=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape_1Const*+
_class!
loc:@loss/conv2d_10_loss/Mean*
valueB *
dtype0*
_output_shapes
: 
°
Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/range/startConst*+
_class!
loc:@loss/conv2d_10_loss/Mean*
value	B : *
dtype0*
_output_shapes
: 
°
Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/range/deltaConst*+
_class!
loc:@loss/conv2d_10_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
Ű
;training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/rangeRangeAtraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/range/start:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/SizeAtraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/range/delta*

Tidx0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
:
Ż
@training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Fill/valueConst*+
_class!
loc:@loss/conv2d_10_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
Ł
:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/FillFill=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape_1@training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Fill/value*
_output_shapes
: *
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*

index_type0
 
Ctraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/DynamicStitchDynamicStitch;training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/range9training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/mod;training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Fill*
_output_shapes
:*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
N
Ž
?training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Maximum/yConst*+
_class!
loc:@loss/conv2d_10_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
 
=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/MaximumMaximumCtraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/DynamicStitch?training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Maximum/y*
_output_shapes
:*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean

>training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/floordivFloorDiv;training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Maximum*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
:*
T0
Ţ
=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/ReshapeReshape?training/Adam/gradients/loss/conv2d_10_loss/Mean_1_grad/truedivCtraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/DynamicStitch*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
Tshape0
Ő
:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/TileTile=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Reshape>training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/floordiv*+
_class!
loc:@loss/conv2d_10_loss/Mean*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0
Á
=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape_2Shapeloss/conv2d_10_loss/Abs*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
out_type0*
_output_shapes
:
Â
=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape_3Shapeloss/conv2d_10_loss/Mean*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
out_type0*
_output_shapes
:
˛
;training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/ConstConst*+
_class!
loc:@loss/conv2d_10_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
Š
:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/ProdProd=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape_2;training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
: 
´
=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Const_1Const*+
_class!
loc:@loss/conv2d_10_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
­
<training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Prod_1Prod=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Shape_3=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Const_1*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
: *
	keep_dims( *

Tidx0
°
Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Maximum_1/yConst*+
_class!
loc:@loss/conv2d_10_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

?training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Maximum_1Maximum<training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Prod_1Atraining/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Maximum_1/y*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
: 

@training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/floordiv_1FloorDiv:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Prod?training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Maximum_1*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*
_output_shapes
: 
ń
:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/CastCast@training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/floordiv_1*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0*+
_class!
loc:@loss/conv2d_10_loss/Mean
Š
=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/truedivRealDiv:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Tile:training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/Cast*
T0*+
_class!
loc:@loss/conv2d_10_loss/Mean*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
Â
9training/Adam/gradients/loss/conv2d_10_loss/Abs_grad/SignSignloss/conv2d_10_loss/sub*
T0**
_class 
loc:@loss/conv2d_10_loss/Abs*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
Ą
8training/Adam/gradients/loss/conv2d_10_loss/Abs_grad/mulMul=training/Adam/gradients/loss/conv2d_10_loss/Mean_grad/truediv9training/Adam/gradients/loss/conv2d_10_loss/Abs_grad/Sign*
T0**
_class 
loc:@loss/conv2d_10_loss/Abs*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ
ˇ
:training/Adam/gradients/loss/conv2d_10_loss/sub_grad/ShapeShapeconv2d_10/Sigmoid*
T0**
_class 
loc:@loss/conv2d_10_loss/sub*
out_type0*
_output_shapes
:
¸
<training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Shape_1Shapeconv2d_10_target*
T0**
_class 
loc:@loss/conv2d_10_loss/sub*
out_type0*
_output_shapes
:
Ć
Jtraining/Adam/gradients/loss/conv2d_10_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs:training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Shape<training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Shape_1*
T0**
_class 
loc:@loss/conv2d_10_loss/sub*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ą
8training/Adam/gradients/loss/conv2d_10_loss/sub_grad/SumSum8training/Adam/gradients/loss/conv2d_10_loss/Abs_grad/mulJtraining/Adam/gradients/loss/conv2d_10_loss/sub_grad/BroadcastGradientArgs**
_class 
loc:@loss/conv2d_10_loss/sub*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ł
<training/Adam/gradients/loss/conv2d_10_loss/sub_grad/ReshapeReshape8training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Sum:training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Shape**
_class 
loc:@loss/conv2d_10_loss/sub*
Tshape0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
T0
ľ
:training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Sum_1Sum8training/Adam/gradients/loss/conv2d_10_loss/Abs_grad/mulLtraining/Adam/gradients/loss/conv2d_10_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0**
_class 
loc:@loss/conv2d_10_loss/sub*
_output_shapes
:
Ę
8training/Adam/gradients/loss/conv2d_10_loss/sub_grad/NegNeg:training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Sum_1*
_output_shapes
:*
T0**
_class 
loc:@loss/conv2d_10_loss/sub
Đ
>training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Reshape_1Reshape8training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Neg<training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Shape_1**
_class 
loc:@loss/conv2d_10_loss/sub*
Tshape0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
ü
:training/Adam/gradients/conv2d_10/Sigmoid_grad/SigmoidGradSigmoidGradconv2d_10/Sigmoid<training/Adam/gradients/loss/conv2d_10_loss/sub_grad/Reshape*$
_class
loc:@conv2d_10/Sigmoid*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ*
T0
ç
:training/Adam/gradients/conv2d_10/BiasAdd_grad/BiasAddGradBiasAddGrad:training/Adam/gradients/conv2d_10/Sigmoid_grad/SigmoidGrad*
T0*$
_class
loc:@conv2d_10/BiasAdd*
data_formatNHWC*
_output_shapes
:
×
9training/Adam/gradients/conv2d_10/convolution_grad/ShapeNShapeNconv2d_9/Reluconv2d_10/kernel/read* 
_output_shapes
::*
T0*(
_class
loc:@conv2d_10/convolution*
out_type0*
N
Ŕ
Ftraining/Adam/gradients/conv2d_10/convolution_grad/Conv2DBackpropInputConv2DBackpropInput9training/Adam/gradients/conv2d_10/convolution_grad/ShapeNconv2d_10/kernel/read:training/Adam/gradients/conv2d_10/Sigmoid_grad/SigmoidGrad*
	dilations
*
T0*(
_class
loc:@conv2d_10/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
ą
Gtraining/Adam/gradients/conv2d_10/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_9/Relu;training/Adam/gradients/conv2d_10/convolution_grad/ShapeN:1:training/Adam/gradients/conv2d_10/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:@*
	dilations
*
T0*(
_class
loc:@conv2d_10/convolution
ô
3training/Adam/gradients/conv2d_9/Relu_grad/ReluGradReluGradFtraining/Adam/gradients/conv2d_10/convolution_grad/Conv2DBackpropInputconv2d_9/Relu*
T0* 
_class
loc:@conv2d_9/Relu*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
Ţ
9training/Adam/gradients/conv2d_9/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_9/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_9/BiasAdd*
data_formatNHWC*
_output_shapes
:@
Ô
8training/Adam/gradients/conv2d_9/convolution_grad/ShapeNShapeNconv2d_8/Reluconv2d_9/kernel/read*
T0*'
_class
loc:@conv2d_9/convolution*
out_type0*
N* 
_output_shapes
::
´
Etraining/Adam/gradients/conv2d_9/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_9/convolution_grad/ShapeNconv2d_9/kernel/read3training/Adam/gradients/conv2d_9/Relu_grad/ReluGrad*'
_class
loc:@conv2d_9/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
	dilations
*
T0
Ś
Ftraining/Adam/gradients/conv2d_9/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_8/Relu:training/Adam/gradients/conv2d_9/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_9/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_9/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@@
ó
3training/Adam/gradients/conv2d_8/Relu_grad/ReluGradReluGradEtraining/Adam/gradients/conv2d_9/convolution_grad/Conv2DBackpropInputconv2d_8/Relu*
T0* 
_class
loc:@conv2d_8/Relu*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
Ţ
9training/Adam/gradients/conv2d_8/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_8/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_8/BiasAdd*
data_formatNHWC*
_output_shapes
:@
Ű
8training/Adam/gradients/conv2d_8/convolution_grad/ShapeNShapeNconcatenate_4/concatconv2d_8/kernel/read*
T0*'
_class
loc:@conv2d_8/convolution*
out_type0*
N* 
_output_shapes
::
ľ
Etraining/Adam/gradients/conv2d_8/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_8/convolution_grad/ShapeNconv2d_8/kernel/read3training/Adam/gradients/conv2d_8/Relu_grad/ReluGrad*'
_class
loc:@conv2d_8/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¸Đ*
	dilations
*
T0
Ž
Ftraining/Adam/gradients/conv2d_8/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconcatenate_4/concat:training/Adam/gradients/conv2d_8/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_8/Relu_grad/ReluGrad*
T0*'
_class
loc:@conv2d_8/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations

Ą
6training/Adam/gradients/concatenate_4/concat_grad/RankConst*'
_class
loc:@concatenate_4/concat*
value	B :*
dtype0*
_output_shapes
: 
Ţ
5training/Adam/gradients/concatenate_4/concat_grad/modFloorModconcatenate_4/concat/axis6training/Adam/gradients/concatenate_4/concat_grad/Rank*
T0*'
_class
loc:@concatenate_4/concat*
_output_shapes
: 
Ť
7training/Adam/gradients/concatenate_4/concat_grad/ShapeShapeconv1b/Relu*
T0*'
_class
loc:@concatenate_4/concat*
out_type0*
_output_shapes
:
Ë
8training/Adam/gradients/concatenate_4/concat_grad/ShapeNShapeNconv1b/Reluconv2d_7/Relu* 
_output_shapes
::*
T0*'
_class
loc:@concatenate_4/concat*
out_type0*
N
Ď
>training/Adam/gradients/concatenate_4/concat_grad/ConcatOffsetConcatOffset5training/Adam/gradients/concatenate_4/concat_grad/mod8training/Adam/gradients/concatenate_4/concat_grad/ShapeN:training/Adam/gradients/concatenate_4/concat_grad/ShapeN:1*'
_class
loc:@concatenate_4/concat*
N* 
_output_shapes
::
ó
7training/Adam/gradients/concatenate_4/concat_grad/SliceSliceEtraining/Adam/gradients/conv2d_8/convolution_grad/Conv2DBackpropInput>training/Adam/gradients/concatenate_4/concat_grad/ConcatOffset8training/Adam/gradients/concatenate_4/concat_grad/ShapeN*
Index0*
T0*'
_class
loc:@concatenate_4/concat*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
ů
9training/Adam/gradients/concatenate_4/concat_grad/Slice_1SliceEtraining/Adam/gradients/conv2d_8/convolution_grad/Conv2DBackpropInput@training/Adam/gradients/concatenate_4/concat_grad/ConcatOffset:1:training/Adam/gradients/concatenate_4/concat_grad/ShapeN:1*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
Index0*
T0*'
_class
loc:@concatenate_4/concat
ç
3training/Adam/gradients/conv2d_7/Relu_grad/ReluGradReluGrad9training/Adam/gradients/concatenate_4/concat_grad/Slice_1conv2d_7/Relu*
T0* 
_class
loc:@conv2d_7/Relu*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
Ţ
9training/Adam/gradients/conv2d_7/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_7/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_7/BiasAdd*
data_formatNHWC*
_output_shapes
:@
ě
8training/Adam/gradients/conv2d_7/convolution_grad/ShapeNShapeN%up_sampling2d_3/ResizeNearestNeighborconv2d_7/kernel/read*
T0*'
_class
loc:@conv2d_7/convolution*
out_type0*
N* 
_output_shapes
::
ľ
Etraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_7/convolution_grad/ShapeNconv2d_7/kernel/read3training/Adam/gradients/conv2d_7/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_7/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¸Đ
ż
Ftraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_3/ResizeNearestNeighbor:training/Adam/gradients/conv2d_7/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_7/Relu_grad/ReluGrad*'
_class
loc:@conv2d_7/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0
ě
atraining/Adam/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@up_sampling2d_3/ResizeNearestNeighbor*
valueB"  (  
Ż
\training/Adam/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradEtraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropInputatraining/Adam/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
align_corners( *
T0*8
_class.
,*loc:@up_sampling2d_3/ResizeNearestNeighbor*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨

3training/Adam/gradients/conv2d_6/Relu_grad/ReluGradReluGrad\training/Adam/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradconv2d_6/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0* 
_class
loc:@conv2d_6/Relu
ß
9training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_6/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_6/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ô
8training/Adam/gradients/conv2d_6/convolution_grad/ShapeNShapeNconv2d_5/Reluconv2d_6/kernel/read*'
_class
loc:@conv2d_6/convolution*
out_type0*
N* 
_output_shapes
::*
T0
ľ
Etraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_6/convolution_grad/ShapeNconv2d_6/kernel/read3training/Adam/gradients/conv2d_6/Relu_grad/ReluGrad*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
	dilations
*
T0*'
_class
loc:@conv2d_6/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
¨
Ftraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_5/Relu:training/Adam/gradients/conv2d_6/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_6/Relu_grad/ReluGrad*(
_output_shapes
:*
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
ô
3training/Adam/gradients/conv2d_5/Relu_grad/ReluGradReluGradEtraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputconv2d_5/Relu*
T0* 
_class
loc:@conv2d_5/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
ß
9training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_5/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_5/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ű
8training/Adam/gradients/conv2d_5/convolution_grad/ShapeNShapeNconcatenate_3/concatconv2d_5/kernel/read*
T0*'
_class
loc:@conv2d_5/convolution*
out_type0*
N* 
_output_shapes
::
ľ
Etraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_5/convolution_grad/ShapeNconv2d_5/kernel/read3training/Adam/gradients/conv2d_5/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_5/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
Ż
Ftraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconcatenate_3/concat:training/Adam/gradients/conv2d_5/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_5/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_5/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:
Ą
6training/Adam/gradients/concatenate_3/concat_grad/RankConst*'
_class
loc:@concatenate_3/concat*
value	B :*
dtype0*
_output_shapes
: 
Ţ
5training/Adam/gradients/concatenate_3/concat_grad/modFloorModconcatenate_3/concat/axis6training/Adam/gradients/concatenate_3/concat_grad/Rank*
T0*'
_class
loc:@concatenate_3/concat*
_output_shapes
: 
Ť
7training/Adam/gradients/concatenate_3/concat_grad/ShapeShapeconv2b/Relu*'
_class
loc:@concatenate_3/concat*
out_type0*
_output_shapes
:*
T0
Ë
8training/Adam/gradients/concatenate_3/concat_grad/ShapeNShapeNconv2b/Reluconv2d_4/Relu*
T0*'
_class
loc:@concatenate_3/concat*
out_type0*
N* 
_output_shapes
::
Ď
>training/Adam/gradients/concatenate_3/concat_grad/ConcatOffsetConcatOffset5training/Adam/gradients/concatenate_3/concat_grad/mod8training/Adam/gradients/concatenate_3/concat_grad/ShapeN:training/Adam/gradients/concatenate_3/concat_grad/ShapeN:1* 
_output_shapes
::*'
_class
loc:@concatenate_3/concat*
N
ô
7training/Adam/gradients/concatenate_3/concat_grad/SliceSliceEtraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropInput>training/Adam/gradients/concatenate_3/concat_grad/ConcatOffset8training/Adam/gradients/concatenate_3/concat_grad/ShapeN*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
Index0*
T0*'
_class
loc:@concatenate_3/concat
ú
9training/Adam/gradients/concatenate_3/concat_grad/Slice_1SliceEtraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropInput@training/Adam/gradients/concatenate_3/concat_grad/ConcatOffset:1:training/Adam/gradients/concatenate_3/concat_grad/ShapeN:1*
Index0*
T0*'
_class
loc:@concatenate_3/concat*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
č
3training/Adam/gradients/conv2d_4/Relu_grad/ReluGradReluGrad9training/Adam/gradients/concatenate_3/concat_grad/Slice_1conv2d_4/Relu* 
_class
loc:@conv2d_4/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0
ß
9training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_4/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_4/BiasAdd*
data_formatNHWC*
_output_shapes	
:
ě
8training/Adam/gradients/conv2d_4/convolution_grad/ShapeNShapeN%up_sampling2d_2/ResizeNearestNeighborconv2d_4/kernel/read* 
_output_shapes
::*
T0*'
_class
loc:@conv2d_4/convolution*
out_type0*
N
ľ
Etraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_4/convolution_grad/ShapeNconv2d_4/kernel/read3training/Adam/gradients/conv2d_4/Relu_grad/ReluGrad*
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
:˙˙˙˙˙˙˙˙˙¨
Ŕ
Ftraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_2/ResizeNearestNeighbor:training/Adam/gradients/conv2d_4/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_4/Relu_grad/ReluGrad*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0*'
_class
loc:@conv2d_4/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
ě
atraining/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor*
valueB"    
Ż
\training/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradEtraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputatraining/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
align_corners( *
T0*8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙

3training/Adam/gradients/conv2d_3/Relu_grad/ReluGradReluGrad\training/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradconv2d_3/Relu*
T0* 
_class
loc:@conv2d_3/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
ß
9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_3/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_3/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ô
8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNShapeNconv2d_2/Reluconv2d_3/kernel/read*
T0*'
_class
loc:@conv2d_3/convolution*
out_type0*
N* 
_output_shapes
::
ľ
Etraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_3/convolution_grad/ShapeNconv2d_3/kernel/read3training/Adam/gradients/conv2d_3/Relu_grad/ReluGrad*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*'
_class
loc:@conv2d_3/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
¨
Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_2/Relu:training/Adam/gradients/conv2d_3/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_3/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_3/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:
ô
3training/Adam/gradients/conv2d_2/Relu_grad/ReluGradReluGradEtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputconv2d_2/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0* 
_class
loc:@conv2d_2/Relu
ß
9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_2/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_2/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ű
8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNShapeNconcatenate_2/concatconv2d_2/kernel/read*
T0*'
_class
loc:@conv2d_2/convolution*
out_type0*
N* 
_output_shapes
::
ľ
Etraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNconv2d_2/kernel/read3training/Adam/gradients/conv2d_2/Relu_grad/ReluGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
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
Ż
Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconcatenate_2/concat:training/Adam/gradients/conv2d_2/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_2/Relu_grad/ReluGrad*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ą
6training/Adam/gradients/concatenate_2/concat_grad/RankConst*
_output_shapes
: *'
_class
loc:@concatenate_2/concat*
value	B :*
dtype0
Ţ
5training/Adam/gradients/concatenate_2/concat_grad/modFloorModconcatenate_2/concat/axis6training/Adam/gradients/concatenate_2/concat_grad/Rank*
T0*'
_class
loc:@concatenate_2/concat*
_output_shapes
: 
Ť
7training/Adam/gradients/concatenate_2/concat_grad/ShapeShapeconv3b/Relu*
T0*'
_class
loc:@concatenate_2/concat*
out_type0*
_output_shapes
:
Ë
8training/Adam/gradients/concatenate_2/concat_grad/ShapeNShapeNconv3b/Reluconv2d_1/Relu* 
_output_shapes
::*
T0*'
_class
loc:@concatenate_2/concat*
out_type0*
N
Ď
>training/Adam/gradients/concatenate_2/concat_grad/ConcatOffsetConcatOffset5training/Adam/gradients/concatenate_2/concat_grad/mod8training/Adam/gradients/concatenate_2/concat_grad/ShapeN:training/Adam/gradients/concatenate_2/concat_grad/ShapeN:1*'
_class
loc:@concatenate_2/concat*
N* 
_output_shapes
::
ô
7training/Adam/gradients/concatenate_2/concat_grad/SliceSliceEtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput>training/Adam/gradients/concatenate_2/concat_grad/ConcatOffset8training/Adam/gradients/concatenate_2/concat_grad/ShapeN*
Index0*
T0*'
_class
loc:@concatenate_2/concat*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
ú
9training/Adam/gradients/concatenate_2/concat_grad/Slice_1SliceEtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput@training/Adam/gradients/concatenate_2/concat_grad/ConcatOffset:1:training/Adam/gradients/concatenate_2/concat_grad/ShapeN:1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
Index0*
T0*'
_class
loc:@concatenate_2/concat
č
3training/Adam/gradients/conv2d_1/Relu_grad/ReluGradReluGrad9training/Adam/gradients/concatenate_2/concat_grad/Slice_1conv2d_1/Relu*
T0* 
_class
loc:@conv2d_1/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
ß
9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_1/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_1/BiasAdd*
data_formatNHWC*
_output_shapes	
:
ě
8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNShapeN%up_sampling2d_1/ResizeNearestNeighborconv2d_1/kernel/read*
T0*'
_class
loc:@conv2d_1/convolution*
out_type0*
N* 
_output_shapes
::
ľ
Etraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNconv2d_1/kernel/read3training/Adam/gradients/conv2d_1/Relu_grad/ReluGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
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
Ŕ
Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_1/ResizeNearestNeighbor:training/Adam/gradients/conv2d_1/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_1/Relu_grad/ReluGrad*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
ě
atraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*
valueB"   Ę   *
dtype0*
_output_shapes
:
Ż
\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradEtraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputatraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
align_corners( *
T0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

1training/Adam/gradients/conv7b/Relu_grad/ReluGradReluGrad\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradconv7b/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0*
_class
loc:@conv7b/Relu
Ů
7training/Adam/gradients/conv7b/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv7b/Relu_grad/ReluGrad*
T0*!
_class
loc:@conv7b/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ě
6training/Adam/gradients/conv7b/convolution_grad/ShapeNShapeNconv7a/Reluconv7b/kernel/read*%
_class
loc:@conv7b/convolution*
out_type0*
N* 
_output_shapes
::*
T0
Ť
Ctraining/Adam/gradients/conv7b/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv7b/convolution_grad/ShapeNconv7b/kernel/read1training/Adam/gradients/conv7b/Relu_grad/ReluGrad*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations
*
T0*%
_class
loc:@conv7b/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

Dtraining/Adam/gradients/conv7b/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv7a/Relu8training/Adam/gradients/conv7b/convolution_grad/ShapeN:11training/Adam/gradients/conv7b/Relu_grad/ReluGrad*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0*%
_class
loc:@conv7b/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
ě
1training/Adam/gradients/conv7a/Relu_grad/ReluGradReluGradCtraining/Adam/gradients/conv7b/convolution_grad/Conv2DBackpropInputconv7a/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0*
_class
loc:@conv7a/Relu
Ů
7training/Adam/gradients/conv7a/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv7a/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*!
_class
loc:@conv7a/BiasAdd*
data_formatNHWC
Ő
6training/Adam/gradients/conv7a/convolution_grad/ShapeNShapeNconcatenate_1/concatconv7a/kernel/read*
T0*%
_class
loc:@conv7a/convolution*
out_type0*
N* 
_output_shapes
::
Ť
Ctraining/Adam/gradients/conv7a/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv7a/convolution_grad/ShapeNconv7a/kernel/read1training/Adam/gradients/conv7a/Relu_grad/ReluGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations
*
T0*%
_class
loc:@conv7a/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
§
Dtraining/Adam/gradients/conv7a/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconcatenate_1/concat8training/Adam/gradients/conv7a/convolution_grad/ShapeN:11training/Adam/gradients/conv7a/Relu_grad/ReluGrad*(
_output_shapes
:*
	dilations
*
T0*%
_class
loc:@conv7a/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ą
6training/Adam/gradients/concatenate_1/concat_grad/RankConst*
_output_shapes
: *'
_class
loc:@concatenate_1/concat*
value	B :*
dtype0
Ţ
5training/Adam/gradients/concatenate_1/concat_grad/modFloorModconcatenate_1/concat/axis6training/Adam/gradients/concatenate_1/concat_grad/Rank*
_output_shapes
: *
T0*'
_class
loc:@concatenate_1/concat
°
7training/Adam/gradients/concatenate_1/concat_grad/ShapeShapedrop4/cond/Merge*
T0*'
_class
loc:@concatenate_1/concat*
out_type0*
_output_shapes
:
×
8training/Adam/gradients/concatenate_1/concat_grad/ShapeNShapeNdrop4/cond/Mergezero_padding2d_1/Pad*
T0*'
_class
loc:@concatenate_1/concat*
out_type0*
N* 
_output_shapes
::
Ď
>training/Adam/gradients/concatenate_1/concat_grad/ConcatOffsetConcatOffset5training/Adam/gradients/concatenate_1/concat_grad/mod8training/Adam/gradients/concatenate_1/concat_grad/ShapeN:training/Adam/gradients/concatenate_1/concat_grad/ShapeN:1* 
_output_shapes
::*'
_class
loc:@concatenate_1/concat*
N
ň
7training/Adam/gradients/concatenate_1/concat_grad/SliceSliceCtraining/Adam/gradients/conv7a/convolution_grad/Conv2DBackpropInput>training/Adam/gradients/concatenate_1/concat_grad/ConcatOffset8training/Adam/gradients/concatenate_1/concat_grad/ShapeN*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
Index0*
T0*'
_class
loc:@concatenate_1/concat
ř
9training/Adam/gradients/concatenate_1/concat_grad/Slice_1SliceCtraining/Adam/gradients/conv7a/convolution_grad/Conv2DBackpropInput@training/Adam/gradients/concatenate_1/concat_grad/ConcatOffset:1:training/Adam/gradients/concatenate_1/concat_grad/ShapeN:1*
Index0*
T0*'
_class
loc:@concatenate_1/concat*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
Ą
6training/Adam/gradients/zero_padding2d_1/Pad_grad/RankConst*
_output_shapes
: *'
_class
loc:@zero_padding2d_1/Pad*
value	B :*
dtype0
¤
9training/Adam/gradients/zero_padding2d_1/Pad_grad/stack/1Const*
dtype0*
_output_shapes
: *'
_class
loc:@zero_padding2d_1/Pad*
value	B :

7training/Adam/gradients/zero_padding2d_1/Pad_grad/stackPack6training/Adam/gradients/zero_padding2d_1/Pad_grad/Rank9training/Adam/gradients/zero_padding2d_1/Pad_grad/stack/1*
N*
_output_shapes
:*
T0*'
_class
loc:@zero_padding2d_1/Pad*

axis 
ˇ
=training/Adam/gradients/zero_padding2d_1/Pad_grad/Slice/beginConst*'
_class
loc:@zero_padding2d_1/Pad*
valueB"        *
dtype0*
_output_shapes
:
ś
7training/Adam/gradients/zero_padding2d_1/Pad_grad/SliceSlicezero_padding2d_1/Pad/paddings=training/Adam/gradients/zero_padding2d_1/Pad_grad/Slice/begin7training/Adam/gradients/zero_padding2d_1/Pad_grad/stack*
Index0*
T0*'
_class
loc:@zero_padding2d_1/Pad*
_output_shapes

:
ť
?training/Adam/gradients/zero_padding2d_1/Pad_grad/Reshape/shapeConst*'
_class
loc:@zero_padding2d_1/Pad*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

9training/Adam/gradients/zero_padding2d_1/Pad_grad/ReshapeReshape7training/Adam/gradients/zero_padding2d_1/Pad_grad/Slice?training/Adam/gradients/zero_padding2d_1/Pad_grad/Reshape/shape*
T0*'
_class
loc:@zero_padding2d_1/Pad*
Tshape0*
_output_shapes
:
Ş
7training/Adam/gradients/zero_padding2d_1/Pad_grad/ShapeShape
conv6/Relu*
T0*'
_class
loc:@zero_padding2d_1/Pad*
out_type0*
_output_shapes
:
ä
9training/Adam/gradients/zero_padding2d_1/Pad_grad/Slice_1Slice9training/Adam/gradients/concatenate_1/concat_grad/Slice_19training/Adam/gradients/zero_padding2d_1/Pad_grad/Reshape7training/Adam/gradients/zero_padding2d_1/Pad_grad/Shape*'
_class
loc:@zero_padding2d_1/Pad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
Index0*
T0
ß
0training/Adam/gradients/conv6/Relu_grad/ReluGradReluGrad9training/Adam/gradients/zero_padding2d_1/Pad_grad/Slice_1
conv6/Relu*
T0*
_class
loc:@conv6/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
Ö
6training/Adam/gradients/conv6/BiasAdd_grad/BiasAddGradBiasAddGrad0training/Adam/gradients/conv6/Relu_grad/ReluGrad*
_output_shapes	
:*
T0* 
_class
loc:@conv6/BiasAdd*
data_formatNHWC
×
5training/Adam/gradients/conv6/convolution_grad/ShapeNShapeNup1/ResizeNearestNeighborconv6/kernel/read*
T0*$
_class
loc:@conv6/convolution*
out_type0*
N* 
_output_shapes
::
Ś
Btraining/Adam/gradients/conv6/convolution_grad/Conv2DBackpropInputConv2DBackpropInput5training/Adam/gradients/conv6/convolution_grad/ShapeNconv6/kernel/read0training/Adam/gradients/conv6/Relu_grad/ReluGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations
*
T0*$
_class
loc:@conv6/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
¨
Ctraining/Adam/gradients/conv6/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterup1/ResizeNearestNeighbor7training/Adam/gradients/conv6/convolution_grad/ShapeN:10training/Adam/gradients/conv6/Relu_grad/ReluGrad*
	dilations
*
T0*$
_class
loc:@conv6/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:
Ô
Utraining/Adam/gradients/up1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
_output_shapes
:*,
_class"
 loc:@up1/ResizeNearestNeighbor*
valueB"C   e   *
dtype0

Ptraining/Adam/gradients/up1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradBtraining/Adam/gradients/conv6/convolution_grad/Conv2DBackpropInputUtraining/Adam/gradients/up1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
align_corners( *
T0*,
_class"
 loc:@up1/ResizeNearestNeighbor*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
Ź
7training/Adam/gradients/drop5/cond/Merge_grad/cond_gradSwitchPtraining/Adam/gradients/up1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGraddrop5/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙Ce:˙˙˙˙˙˙˙˙˙Ce*
T0*,
_class"
 loc:@up1/ResizeNearestNeighbor
Ŕ
training/Adam/gradients/SwitchSwitchconv5b/Reludrop5/cond/pred_id*
T0*
_class
loc:@conv5b/Relu*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙Ce:˙˙˙˙˙˙˙˙˙Ce
Š
 training/Adam/gradients/IdentityIdentity training/Adam/gradients/Switch:1*
T0*
_class
loc:@conv5b/Relu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

training/Adam/gradients/Shape_1Shape training/Adam/gradients/Switch:1*
_class
loc:@conv5b/Relu*
out_type0*
_output_shapes
:*
T0
Ť
#training/Adam/gradients/zeros/ConstConst!^training/Adam/gradients/Identity*
_class
loc:@conv5b/Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
Ř
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*
T0*
_class
loc:@conv5b/Relu*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

:training/Adam/gradients/drop5/cond/Switch_1_grad/cond_gradMerge7training/Adam/gradients/drop5/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros*
T0*
_class
loc:@conv5b/Relu*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ce: 
ž
9training/Adam/gradients/drop5/cond/dropout/mul_grad/ShapeShapedrop5/cond/dropout/truediv*
_output_shapes
:*
T0*)
_class
loc:@drop5/cond/dropout/mul*
out_type0
ž
;training/Adam/gradients/drop5/cond/dropout/mul_grad/Shape_1Shapedrop5/cond/dropout/Floor*
_output_shapes
:*
T0*)
_class
loc:@drop5/cond/dropout/mul*
out_type0
Â
Itraining/Adam/gradients/drop5/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/drop5/cond/dropout/mul_grad/Shape;training/Adam/gradients/drop5/cond/dropout/mul_grad/Shape_1*
T0*)
_class
loc:@drop5/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ů
7training/Adam/gradients/drop5/cond/dropout/mul_grad/MulMul9training/Adam/gradients/drop5/cond/Merge_grad/cond_grad:1drop5/cond/dropout/Floor*
T0*)
_class
loc:@drop5/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
­
7training/Adam/gradients/drop5/cond/dropout/mul_grad/SumSum7training/Adam/gradients/drop5/cond/dropout/mul_grad/MulItraining/Adam/gradients/drop5/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*)
_class
loc:@drop5/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Ž
;training/Adam/gradients/drop5/cond/dropout/mul_grad/ReshapeReshape7training/Adam/gradients/drop5/cond/dropout/mul_grad/Sum9training/Adam/gradients/drop5/cond/dropout/mul_grad/Shape*
T0*)
_class
loc:@drop5/cond/dropout/mul*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
ý
9training/Adam/gradients/drop5/cond/dropout/mul_grad/Mul_1Muldrop5/cond/dropout/truediv9training/Adam/gradients/drop5/cond/Merge_grad/cond_grad:1*
T0*)
_class
loc:@drop5/cond/dropout/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
ł
9training/Adam/gradients/drop5/cond/dropout/mul_grad/Sum_1Sum9training/Adam/gradients/drop5/cond/dropout/mul_grad/Mul_1Ktraining/Adam/gradients/drop5/cond/dropout/mul_grad/BroadcastGradientArgs:1*
T0*)
_class
loc:@drop5/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
´
=training/Adam/gradients/drop5/cond/dropout/mul_grad/Reshape_1Reshape9training/Adam/gradients/drop5/cond/dropout/mul_grad/Sum_1;training/Adam/gradients/drop5/cond/dropout/mul_grad/Shape_1*
T0*)
_class
loc:@drop5/cond/dropout/mul*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
ş
=training/Adam/gradients/drop5/cond/dropout/truediv_grad/ShapeShapedrop5/cond/mul*
_output_shapes
:*
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*
out_type0
ą
?training/Adam/gradients/drop5/cond/dropout/truediv_grad/Shape_1Const*-
_class#
!loc:@drop5/cond/dropout/truediv*
valueB *
dtype0*
_output_shapes
: 
Ň
Mtraining/Adam/gradients/drop5/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/drop5/cond/dropout/truediv_grad/Shape?training/Adam/gradients/drop5/cond/dropout/truediv_grad/Shape_1*
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

?training/Adam/gradients/drop5/cond/dropout/truediv_grad/RealDivRealDiv;training/Adam/gradients/drop5/cond/dropout/mul_grad/Reshapedrop5/cond/dropout/sub*
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
Á
;training/Adam/gradients/drop5/cond/dropout/truediv_grad/SumSum?training/Adam/gradients/drop5/cond/dropout/truediv_grad/RealDivMtraining/Adam/gradients/drop5/cond/dropout/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*
_output_shapes
:
ž
?training/Adam/gradients/drop5/cond/dropout/truediv_grad/ReshapeReshape;training/Adam/gradients/drop5/cond/dropout/truediv_grad/Sum=training/Adam/gradients/drop5/cond/dropout/truediv_grad/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*
Tshape0
ź
;training/Adam/gradients/drop5/cond/dropout/truediv_grad/NegNegdrop5/cond/mul*
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

Atraining/Adam/gradients/drop5/cond/dropout/truediv_grad/RealDiv_1RealDiv;training/Adam/gradients/drop5/cond/dropout/truediv_grad/Negdrop5/cond/dropout/sub*
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

Atraining/Adam/gradients/drop5/cond/dropout/truediv_grad/RealDiv_2RealDivAtraining/Adam/gradients/drop5/cond/dropout/truediv_grad/RealDiv_1drop5/cond/dropout/sub*
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
Ź
;training/Adam/gradients/drop5/cond/dropout/truediv_grad/mulMul;training/Adam/gradients/drop5/cond/dropout/mul_grad/ReshapeAtraining/Adam/gradients/drop5/cond/dropout/truediv_grad/RealDiv_2*
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
Á
=training/Adam/gradients/drop5/cond/dropout/truediv_grad/Sum_1Sum;training/Adam/gradients/drop5/cond/dropout/truediv_grad/mulOtraining/Adam/gradients/drop5/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0
Ş
Atraining/Adam/gradients/drop5/cond/dropout/truediv_grad/Reshape_1Reshape=training/Adam/gradients/drop5/cond/dropout/truediv_grad/Sum_1?training/Adam/gradients/drop5/cond/dropout/truediv_grad/Shape_1*
_output_shapes
: *
T0*-
_class#
!loc:@drop5/cond/dropout/truediv*
Tshape0
Ť
1training/Adam/gradients/drop5/cond/mul_grad/ShapeShapedrop5/cond/mul/Switch:1*
T0*!
_class
loc:@drop5/cond/mul*
out_type0*
_output_shapes
:

3training/Adam/gradients/drop5/cond/mul_grad/Shape_1Const*!
_class
loc:@drop5/cond/mul*
valueB *
dtype0*
_output_shapes
: 
˘
Atraining/Adam/gradients/drop5/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1training/Adam/gradients/drop5/cond/mul_grad/Shape3training/Adam/gradients/drop5/cond/mul_grad/Shape_1*
T0*!
_class
loc:@drop5/cond/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ç
/training/Adam/gradients/drop5/cond/mul_grad/MulMul?training/Adam/gradients/drop5/cond/dropout/truediv_grad/Reshapedrop5/cond/mul/y*
T0*!
_class
loc:@drop5/cond/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

/training/Adam/gradients/drop5/cond/mul_grad/SumSum/training/Adam/gradients/drop5/cond/mul_grad/MulAtraining/Adam/gradients/drop5/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*!
_class
loc:@drop5/cond/mul*
_output_shapes
:

3training/Adam/gradients/drop5/cond/mul_grad/ReshapeReshape/training/Adam/gradients/drop5/cond/mul_grad/Sum1training/Adam/gradients/drop5/cond/mul_grad/Shape*
T0*!
_class
loc:@drop5/cond/mul*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
đ
1training/Adam/gradients/drop5/cond/mul_grad/Mul_1Muldrop5/cond/mul/Switch:1?training/Adam/gradients/drop5/cond/dropout/truediv_grad/Reshape*
T0*!
_class
loc:@drop5/cond/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

1training/Adam/gradients/drop5/cond/mul_grad/Sum_1Sum1training/Adam/gradients/drop5/cond/mul_grad/Mul_1Ctraining/Adam/gradients/drop5/cond/mul_grad/BroadcastGradientArgs:1*!
_class
loc:@drop5/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ú
5training/Adam/gradients/drop5/cond/mul_grad/Reshape_1Reshape1training/Adam/gradients/drop5/cond/mul_grad/Sum_13training/Adam/gradients/drop5/cond/mul_grad/Shape_1*
T0*!
_class
loc:@drop5/cond/mul*
Tshape0*
_output_shapes
: 
Â
 training/Adam/gradients/Switch_1Switchconv5b/Reludrop5/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙Ce:˙˙˙˙˙˙˙˙˙Ce*
T0*
_class
loc:@conv5b/Relu
Ť
"training/Adam/gradients/Identity_1Identity training/Adam/gradients/Switch_1*
T0*
_class
loc:@conv5b/Relu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

training/Adam/gradients/Shape_2Shape training/Adam/gradients/Switch_1*
_class
loc:@conv5b/Relu*
out_type0*
_output_shapes
:*
T0
Ż
%training/Adam/gradients/zeros_1/ConstConst#^training/Adam/gradients/Identity_1*
_class
loc:@conv5b/Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
Ü
training/Adam/gradients/zeros_1Filltraining/Adam/gradients/Shape_2%training/Adam/gradients/zeros_1/Const*
T0*
_class
loc:@conv5b/Relu*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce

<training/Adam/gradients/drop5/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_13training/Adam/gradients/drop5/cond/mul_grad/Reshape*
T0*
_class
loc:@conv5b/Relu*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ce: 

training/Adam/gradients/AddNAddN:training/Adam/gradients/drop5/cond/Switch_1_grad/cond_grad<training/Adam/gradients/drop5/cond/mul/Switch_grad/cond_grad*
T0*
_class
loc:@conv5b/Relu*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
Ă
1training/Adam/gradients/conv5b/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddNconv5b/Relu*
T0*
_class
loc:@conv5b/Relu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
Ů
7training/Adam/gradients/conv5b/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv5b/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*!
_class
loc:@conv5b/BiasAdd*
data_formatNHWC
Ě
6training/Adam/gradients/conv5b/convolution_grad/ShapeNShapeNconv5a/Reluconv5b/kernel/read*
T0*%
_class
loc:@conv5b/convolution*
out_type0*
N* 
_output_shapes
::
Š
Ctraining/Adam/gradients/conv5b/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv5b/convolution_grad/ShapeNconv5b/kernel/read1training/Adam/gradients/conv5b/Relu_grad/ReluGrad*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce*
	dilations
*
T0*%
_class
loc:@conv5b/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

Dtraining/Adam/gradients/conv5b/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv5a/Relu8training/Adam/gradients/conv5b/convolution_grad/ShapeN:11training/Adam/gradients/conv5b/Relu_grad/ReluGrad*
	dilations
*
T0*%
_class
loc:@conv5b/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:
ę
1training/Adam/gradients/conv5a/Relu_grad/ReluGradReluGradCtraining/Adam/gradients/conv5b/convolution_grad/Conv2DBackpropInputconv5a/Relu*
T0*
_class
loc:@conv5a/Relu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
Ů
7training/Adam/gradients/conv5a/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv5a/Relu_grad/ReluGrad*!
_class
loc:@conv5a/BiasAdd*
data_formatNHWC*
_output_shapes	
:*
T0
Î
6training/Adam/gradients/conv5a/convolution_grad/ShapeNShapeNpool4/MaxPoolconv5a/kernel/read*
T0*%
_class
loc:@conv5a/convolution*
out_type0*
N* 
_output_shapes
::
Š
Ctraining/Adam/gradients/conv5a/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv5a/convolution_grad/ShapeNconv5a/kernel/read1training/Adam/gradients/conv5a/Relu_grad/ReluGrad*
	dilations
*
T0*%
_class
loc:@conv5a/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙Ce
 
Dtraining/Adam/gradients/conv5a/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterpool4/MaxPool8training/Adam/gradients/conv5a/convolution_grad/ShapeN:11training/Adam/gradients/conv5a/Relu_grad/ReluGrad*(
_output_shapes
:*
	dilations
*
T0*%
_class
loc:@conv5a/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Ý
6training/Adam/gradients/pool4/MaxPool_grad/MaxPoolGradMaxPoolGraddrop4/cond/Mergepool4/MaxPoolCtraining/Adam/gradients/conv5a/convolution_grad/Conv2DBackpropInput*
ksize
*
paddingVALID*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0* 
_class
loc:@pool4/MaxPool*
strides
*
data_formatNHWC

training/Adam/gradients/AddN_1AddN7training/Adam/gradients/concatenate_1/concat_grad/Slice6training/Adam/gradients/pool4/MaxPool_grad/MaxPoolGrad*
T0*'
_class
loc:@concatenate_1/concat*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
ů
7training/Adam/gradients/drop4/cond/Merge_grad/cond_gradSwitchtraining/Adam/gradients/AddN_1drop4/cond/pred_id*'
_class
loc:@concatenate_1/concat*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙Ę:˙˙˙˙˙˙˙˙˙Ę*
T0
Ć
 training/Adam/gradients/Switch_2Switchconv4b/Reludrop4/cond/pred_id*
T0*
_class
loc:@conv4b/Relu*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙Ę:˙˙˙˙˙˙˙˙˙Ę
Ż
"training/Adam/gradients/Identity_2Identity"training/Adam/gradients/Switch_2:1*
T0*
_class
loc:@conv4b/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
Ą
training/Adam/gradients/Shape_3Shape"training/Adam/gradients/Switch_2:1*
T0*
_class
loc:@conv4b/Relu*
out_type0*
_output_shapes
:
Ż
%training/Adam/gradients/zeros_2/ConstConst#^training/Adam/gradients/Identity_2*
_class
loc:@conv4b/Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
Ţ
training/Adam/gradients/zeros_2Filltraining/Adam/gradients/Shape_3%training/Adam/gradients/zeros_2/Const*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0*
_class
loc:@conv4b/Relu*

index_type0

:training/Adam/gradients/drop4/cond/Switch_1_grad/cond_gradMerge7training/Adam/gradients/drop4/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_2*
T0*
_class
loc:@conv4b/Relu*
N*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙Ę: 
ž
9training/Adam/gradients/drop4/cond/dropout/mul_grad/ShapeShapedrop4/cond/dropout/truediv*
_output_shapes
:*
T0*)
_class
loc:@drop4/cond/dropout/mul*
out_type0
ž
;training/Adam/gradients/drop4/cond/dropout/mul_grad/Shape_1Shapedrop4/cond/dropout/Floor*
_output_shapes
:*
T0*)
_class
loc:@drop4/cond/dropout/mul*
out_type0
Â
Itraining/Adam/gradients/drop4/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/drop4/cond/dropout/mul_grad/Shape;training/Adam/gradients/drop4/cond/dropout/mul_grad/Shape_1*
T0*)
_class
loc:@drop4/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ű
7training/Adam/gradients/drop4/cond/dropout/mul_grad/MulMul9training/Adam/gradients/drop4/cond/Merge_grad/cond_grad:1drop4/cond/dropout/Floor*)
_class
loc:@drop4/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0
­
7training/Adam/gradients/drop4/cond/dropout/mul_grad/SumSum7training/Adam/gradients/drop4/cond/dropout/mul_grad/MulItraining/Adam/gradients/drop4/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*)
_class
loc:@drop4/cond/dropout/mul
°
;training/Adam/gradients/drop4/cond/dropout/mul_grad/ReshapeReshape7training/Adam/gradients/drop4/cond/dropout/mul_grad/Sum9training/Adam/gradients/drop4/cond/dropout/mul_grad/Shape*
T0*)
_class
loc:@drop4/cond/dropout/mul*
Tshape0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
˙
9training/Adam/gradients/drop4/cond/dropout/mul_grad/Mul_1Muldrop4/cond/dropout/truediv9training/Adam/gradients/drop4/cond/Merge_grad/cond_grad:1*
T0*)
_class
loc:@drop4/cond/dropout/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
ł
9training/Adam/gradients/drop4/cond/dropout/mul_grad/Sum_1Sum9training/Adam/gradients/drop4/cond/dropout/mul_grad/Mul_1Ktraining/Adam/gradients/drop4/cond/dropout/mul_grad/BroadcastGradientArgs:1*)
_class
loc:@drop4/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ś
=training/Adam/gradients/drop4/cond/dropout/mul_grad/Reshape_1Reshape9training/Adam/gradients/drop4/cond/dropout/mul_grad/Sum_1;training/Adam/gradients/drop4/cond/dropout/mul_grad/Shape_1*
T0*)
_class
loc:@drop4/cond/dropout/mul*
Tshape0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
ş
=training/Adam/gradients/drop4/cond/dropout/truediv_grad/ShapeShapedrop4/cond/mul*
_output_shapes
:*
T0*-
_class#
!loc:@drop4/cond/dropout/truediv*
out_type0
ą
?training/Adam/gradients/drop4/cond/dropout/truediv_grad/Shape_1Const*-
_class#
!loc:@drop4/cond/dropout/truediv*
valueB *
dtype0*
_output_shapes
: 
Ň
Mtraining/Adam/gradients/drop4/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/drop4/cond/dropout/truediv_grad/Shape?training/Adam/gradients/drop4/cond/dropout/truediv_grad/Shape_1*
T0*-
_class#
!loc:@drop4/cond/dropout/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

?training/Adam/gradients/drop4/cond/dropout/truediv_grad/RealDivRealDiv;training/Adam/gradients/drop4/cond/dropout/mul_grad/Reshapedrop4/cond/dropout/sub*
T0*-
_class#
!loc:@drop4/cond/dropout/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
Á
;training/Adam/gradients/drop4/cond/dropout/truediv_grad/SumSum?training/Adam/gradients/drop4/cond/dropout/truediv_grad/RealDivMtraining/Adam/gradients/drop4/cond/dropout/truediv_grad/BroadcastGradientArgs*-
_class#
!loc:@drop4/cond/dropout/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ŕ
?training/Adam/gradients/drop4/cond/dropout/truediv_grad/ReshapeReshape;training/Adam/gradients/drop4/cond/dropout/truediv_grad/Sum=training/Adam/gradients/drop4/cond/dropout/truediv_grad/Shape*
T0*-
_class#
!loc:@drop4/cond/dropout/truediv*
Tshape0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
ž
;training/Adam/gradients/drop4/cond/dropout/truediv_grad/NegNegdrop4/cond/mul*-
_class#
!loc:@drop4/cond/dropout/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0

Atraining/Adam/gradients/drop4/cond/dropout/truediv_grad/RealDiv_1RealDiv;training/Adam/gradients/drop4/cond/dropout/truediv_grad/Negdrop4/cond/dropout/sub*
T0*-
_class#
!loc:@drop4/cond/dropout/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

Atraining/Adam/gradients/drop4/cond/dropout/truediv_grad/RealDiv_2RealDivAtraining/Adam/gradients/drop4/cond/dropout/truediv_grad/RealDiv_1drop4/cond/dropout/sub*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0*-
_class#
!loc:@drop4/cond/dropout/truediv
Ž
;training/Adam/gradients/drop4/cond/dropout/truediv_grad/mulMul;training/Adam/gradients/drop4/cond/dropout/mul_grad/ReshapeAtraining/Adam/gradients/drop4/cond/dropout/truediv_grad/RealDiv_2*
T0*-
_class#
!loc:@drop4/cond/dropout/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
Á
=training/Adam/gradients/drop4/cond/dropout/truediv_grad/Sum_1Sum;training/Adam/gradients/drop4/cond/dropout/truediv_grad/mulOtraining/Adam/gradients/drop4/cond/dropout/truediv_grad/BroadcastGradientArgs:1*-
_class#
!loc:@drop4/cond/dropout/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ş
Atraining/Adam/gradients/drop4/cond/dropout/truediv_grad/Reshape_1Reshape=training/Adam/gradients/drop4/cond/dropout/truediv_grad/Sum_1?training/Adam/gradients/drop4/cond/dropout/truediv_grad/Shape_1*
T0*-
_class#
!loc:@drop4/cond/dropout/truediv*
Tshape0*
_output_shapes
: 
Ť
1training/Adam/gradients/drop4/cond/mul_grad/ShapeShapedrop4/cond/mul/Switch:1*
T0*!
_class
loc:@drop4/cond/mul*
out_type0*
_output_shapes
:

3training/Adam/gradients/drop4/cond/mul_grad/Shape_1Const*!
_class
loc:@drop4/cond/mul*
valueB *
dtype0*
_output_shapes
: 
˘
Atraining/Adam/gradients/drop4/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1training/Adam/gradients/drop4/cond/mul_grad/Shape3training/Adam/gradients/drop4/cond/mul_grad/Shape_1*
T0*!
_class
loc:@drop4/cond/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
é
/training/Adam/gradients/drop4/cond/mul_grad/MulMul?training/Adam/gradients/drop4/cond/dropout/truediv_grad/Reshapedrop4/cond/mul/y*
T0*!
_class
loc:@drop4/cond/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

/training/Adam/gradients/drop4/cond/mul_grad/SumSum/training/Adam/gradients/drop4/cond/mul_grad/MulAtraining/Adam/gradients/drop4/cond/mul_grad/BroadcastGradientArgs*!
_class
loc:@drop4/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

3training/Adam/gradients/drop4/cond/mul_grad/ReshapeReshape/training/Adam/gradients/drop4/cond/mul_grad/Sum1training/Adam/gradients/drop4/cond/mul_grad/Shape*
T0*!
_class
loc:@drop4/cond/mul*
Tshape0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
ň
1training/Adam/gradients/drop4/cond/mul_grad/Mul_1Muldrop4/cond/mul/Switch:1?training/Adam/gradients/drop4/cond/dropout/truediv_grad/Reshape*
T0*!
_class
loc:@drop4/cond/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

1training/Adam/gradients/drop4/cond/mul_grad/Sum_1Sum1training/Adam/gradients/drop4/cond/mul_grad/Mul_1Ctraining/Adam/gradients/drop4/cond/mul_grad/BroadcastGradientArgs:1*!
_class
loc:@drop4/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ú
5training/Adam/gradients/drop4/cond/mul_grad/Reshape_1Reshape1training/Adam/gradients/drop4/cond/mul_grad/Sum_13training/Adam/gradients/drop4/cond/mul_grad/Shape_1*
T0*!
_class
loc:@drop4/cond/mul*
Tshape0*
_output_shapes
: 
Ć
 training/Adam/gradients/Switch_3Switchconv4b/Reludrop4/cond/pred_id*
T0*
_class
loc:@conv4b/Relu*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙Ę:˙˙˙˙˙˙˙˙˙Ę
­
"training/Adam/gradients/Identity_3Identity training/Adam/gradients/Switch_3*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0*
_class
loc:@conv4b/Relu

training/Adam/gradients/Shape_4Shape training/Adam/gradients/Switch_3*
T0*
_class
loc:@conv4b/Relu*
out_type0*
_output_shapes
:
Ż
%training/Adam/gradients/zeros_3/ConstConst#^training/Adam/gradients/Identity_3*
_class
loc:@conv4b/Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
Ţ
training/Adam/gradients/zeros_3Filltraining/Adam/gradients/Shape_4%training/Adam/gradients/zeros_3/Const*
T0*
_class
loc:@conv4b/Relu*

index_type0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

<training/Adam/gradients/drop4/cond/mul/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_33training/Adam/gradients/drop4/cond/mul_grad/Reshape*
T0*
_class
loc:@conv4b/Relu*
N*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙Ę: 

training/Adam/gradients/AddN_2AddN:training/Adam/gradients/drop4/cond/Switch_1_grad/cond_grad<training/Adam/gradients/drop4/cond/mul/Switch_grad/cond_grad*
T0*
_class
loc:@conv4b/Relu*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
Ç
1training/Adam/gradients/conv4b/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_2conv4b/Relu*
T0*
_class
loc:@conv4b/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę
Ů
7training/Adam/gradients/conv4b/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv4b/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0*!
_class
loc:@conv4b/BiasAdd
Ě
6training/Adam/gradients/conv4b/convolution_grad/ShapeNShapeNconv4a/Reluconv4b/kernel/read*%
_class
loc:@conv4b/convolution*
out_type0*
N* 
_output_shapes
::*
T0
Ť
Ctraining/Adam/gradients/conv4b/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv4b/convolution_grad/ShapeNconv4b/kernel/read1training/Adam/gradients/conv4b/Relu_grad/ReluGrad*
	dilations
*
T0*%
_class
loc:@conv4b/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę

Dtraining/Adam/gradients/conv4b/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv4a/Relu8training/Adam/gradients/conv4b/convolution_grad/ShapeN:11training/Adam/gradients/conv4b/Relu_grad/ReluGrad*
	dilations
*
T0*%
_class
loc:@conv4b/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:
ě
1training/Adam/gradients/conv4a/Relu_grad/ReluGradReluGradCtraining/Adam/gradients/conv4b/convolution_grad/Conv2DBackpropInputconv4a/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
T0*
_class
loc:@conv4a/Relu
Ů
7training/Adam/gradients/conv4a/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv4a/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*!
_class
loc:@conv4a/BiasAdd*
data_formatNHWC
Î
6training/Adam/gradients/conv4a/convolution_grad/ShapeNShapeNpool3/MaxPoolconv4a/kernel/read* 
_output_shapes
::*
T0*%
_class
loc:@conv4a/convolution*
out_type0*
N
Ť
Ctraining/Adam/gradients/conv4a/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv4a/convolution_grad/ShapeNconv4a/kernel/read1training/Adam/gradients/conv4a/Relu_grad/ReluGrad*%
_class
loc:@conv4a/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙Ę*
	dilations
*
T0
 
Dtraining/Adam/gradients/conv4a/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterpool3/MaxPool8training/Adam/gradients/conv4a/convolution_grad/ShapeN:11training/Adam/gradients/conv4a/Relu_grad/ReluGrad*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0*%
_class
loc:@conv4a/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
Ř
6training/Adam/gradients/pool3/MaxPool_grad/MaxPoolGradMaxPoolGradconv3b/Relupool3/MaxPoolCtraining/Adam/gradients/conv4a/convolution_grad/Conv2DBackpropInput*
ksize
*
paddingVALID*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0* 
_class
loc:@pool3/MaxPool*
strides
*
data_formatNHWC

training/Adam/gradients/AddN_3AddN7training/Adam/gradients/concatenate_2/concat_grad/Slice6training/Adam/gradients/pool3/MaxPool_grad/MaxPoolGrad*
T0*'
_class
loc:@concatenate_2/concat*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
Ç
1training/Adam/gradients/conv3b/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_3conv3b/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0*
_class
loc:@conv3b/Relu
Ů
7training/Adam/gradients/conv3b/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv3b/Relu_grad/ReluGrad*
T0*!
_class
loc:@conv3b/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Ě
6training/Adam/gradients/conv3b/convolution_grad/ShapeNShapeNconv3a/Reluconv3b/kernel/read* 
_output_shapes
::*
T0*%
_class
loc:@conv3b/convolution*
out_type0*
N
Ť
Ctraining/Adam/gradients/conv3b/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv3b/convolution_grad/ShapeNconv3b/kernel/read1training/Adam/gradients/conv3b/Relu_grad/ReluGrad*
	dilations
*
T0*%
_class
loc:@conv3b/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙

Dtraining/Adam/gradients/conv3b/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv3a/Relu8training/Adam/gradients/conv3b/convolution_grad/ShapeN:11training/Adam/gradients/conv3b/Relu_grad/ReluGrad*(
_output_shapes
:*
	dilations
*
T0*%
_class
loc:@conv3b/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
ě
1training/Adam/gradients/conv3a/Relu_grad/ReluGradReluGradCtraining/Adam/gradients/conv3b/convolution_grad/Conv2DBackpropInputconv3a/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
T0*
_class
loc:@conv3a/Relu
Ů
7training/Adam/gradients/conv3a/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv3a/Relu_grad/ReluGrad*!
_class
loc:@conv3a/BiasAdd*
data_formatNHWC*
_output_shapes	
:*
T0
Î
6training/Adam/gradients/conv3a/convolution_grad/ShapeNShapeNpool2/MaxPoolconv3a/kernel/read*
N* 
_output_shapes
::*
T0*%
_class
loc:@conv3a/convolution*
out_type0
Ť
Ctraining/Adam/gradients/conv3a/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv3a/convolution_grad/ShapeNconv3a/kernel/read1training/Adam/gradients/conv3a/Relu_grad/ReluGrad*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*%
_class
loc:@conv3a/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
 
Dtraining/Adam/gradients/conv3a/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterpool2/MaxPool8training/Adam/gradients/conv3a/convolution_grad/ShapeN:11training/Adam/gradients/conv3a/Relu_grad/ReluGrad*
	dilations
*
T0*%
_class
loc:@conv3a/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:
Ř
6training/Adam/gradients/pool2/MaxPool_grad/MaxPoolGradMaxPoolGradconv2b/Relupool2/MaxPoolCtraining/Adam/gradients/conv3a/convolution_grad/Conv2DBackpropInput*
ksize
*
paddingVALID*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0* 
_class
loc:@pool2/MaxPool*
strides
*
data_formatNHWC

training/Adam/gradients/AddN_4AddN7training/Adam/gradients/concatenate_3/concat_grad/Slice6training/Adam/gradients/pool2/MaxPool_grad/MaxPoolGrad*'
_class
loc:@concatenate_3/concat*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
T0
Ç
1training/Adam/gradients/conv2b/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_4conv2b/Relu*
T0*
_class
loc:@conv2b/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
Ů
7training/Adam/gradients/conv2b/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv2b/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*!
_class
loc:@conv2b/BiasAdd*
data_formatNHWC
Ě
6training/Adam/gradients/conv2b/convolution_grad/ShapeNShapeNconv2a/Reluconv2b/kernel/read*
T0*%
_class
loc:@conv2b/convolution*
out_type0*
N* 
_output_shapes
::
Ť
Ctraining/Adam/gradients/conv2b/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv2b/convolution_grad/ShapeNconv2b/kernel/read1training/Adam/gradients/conv2b/Relu_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨*
	dilations
*
T0*%
_class
loc:@conv2b/convolution

Dtraining/Adam/gradients/conv2b/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2a/Relu8training/Adam/gradients/conv2b/convolution_grad/ShapeN:11training/Adam/gradients/conv2b/Relu_grad/ReluGrad*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0*%
_class
loc:@conv2b/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
ě
1training/Adam/gradients/conv2a/Relu_grad/ReluGradReluGradCtraining/Adam/gradients/conv2b/convolution_grad/Conv2DBackpropInputconv2a/Relu*
T0*
_class
loc:@conv2a/Relu*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙¨
Ů
7training/Adam/gradients/conv2a/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv2a/Relu_grad/ReluGrad*
T0*!
_class
loc:@conv2a/BiasAdd*
data_formatNHWC*
_output_shapes	
:
Î
6training/Adam/gradients/conv2a/convolution_grad/ShapeNShapeNpool1/MaxPoolconv2a/kernel/read*%
_class
loc:@conv2a/convolution*
out_type0*
N* 
_output_shapes
::*
T0
Ş
Ctraining/Adam/gradients/conv2a/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv2a/convolution_grad/ShapeNconv2a/kernel/read1training/Adam/gradients/conv2a/Relu_grad/ReluGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¨@*
	dilations
*
T0*%
_class
loc:@conv2a/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME

Dtraining/Adam/gradients/conv2a/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterpool1/MaxPool8training/Adam/gradients/conv2a/convolution_grad/ShapeN:11training/Adam/gradients/conv2a/Relu_grad/ReluGrad*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*%
_class
loc:@conv2a/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
×
6training/Adam/gradients/pool1/MaxPool_grad/MaxPoolGradMaxPoolGradconv1b/Relupool1/MaxPoolCtraining/Adam/gradients/conv2a/convolution_grad/Conv2DBackpropInput*
T0* 
_class
loc:@pool1/MaxPool*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@

training/Adam/gradients/AddN_5AddN7training/Adam/gradients/concatenate_4/concat_grad/Slice6training/Adam/gradients/pool1/MaxPool_grad/MaxPoolGrad*
T0*'
_class
loc:@concatenate_4/concat*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
Ć
1training/Adam/gradients/conv1b/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_5conv1b/Relu*
T0*
_class
loc:@conv1b/Relu*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
Ř
7training/Adam/gradients/conv1b/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv1b/Relu_grad/ReluGrad*
T0*!
_class
loc:@conv1b/BiasAdd*
data_formatNHWC*
_output_shapes
:@
Ě
6training/Adam/gradients/conv1b/convolution_grad/ShapeNShapeNconv1a/Reluconv1b/kernel/read* 
_output_shapes
::*
T0*%
_class
loc:@conv1b/convolution*
out_type0*
N
Ş
Ctraining/Adam/gradients/conv1b/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv1b/convolution_grad/ShapeNconv1b/kernel/read1training/Adam/gradients/conv1b/Relu_grad/ReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@*
	dilations
*
T0*%
_class
loc:@conv1b/convolution

Dtraining/Adam/gradients/conv1b/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv1a/Relu8training/Adam/gradients/conv1b/convolution_grad/ShapeN:11training/Adam/gradients/conv1b/Relu_grad/ReluGrad*
paddingSAME*&
_output_shapes
:@@*
	dilations
*
T0*%
_class
loc:@conv1b/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
ë
1training/Adam/gradients/conv1a/Relu_grad/ReluGradReluGradCtraining/Adam/gradients/conv1b/convolution_grad/Conv2DBackpropInputconv1a/Relu*
T0*
_class
loc:@conv1a/Relu*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ@
Ř
7training/Adam/gradients/conv1a/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv1a/Relu_grad/ReluGrad*
T0*!
_class
loc:@conv1a/BiasAdd*
data_formatNHWC*
_output_shapes
:@
Č
6training/Adam/gradients/conv1a/convolution_grad/ShapeNShapeNinput_1conv1a/kernel/read*
T0*%
_class
loc:@conv1a/convolution*
out_type0*
N* 
_output_shapes
::
Ş
Ctraining/Adam/gradients/conv1a/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6training/Adam/gradients/conv1a/convolution_grad/ShapeNconv1a/kernel/read1training/Adam/gradients/conv1a/Relu_grad/ReluGrad*
	dilations
*
T0*%
_class
loc:@conv1a/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙¸Đ

Dtraining/Adam/gradients/conv1a/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterinput_18training/Adam/gradients/conv1a/convolution_grad/ShapeN:11training/Adam/gradients/conv1a/Relu_grad/ReluGrad*
paddingSAME*&
_output_shapes
:@*
	dilations
*
T0*%
_class
loc:@conv1a/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
_
training/Adam/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
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
Truncate( *

DstT0*
_output_shapes
: 
^
training/Adam/mulMulAdam/decay/readtraining/Adam/Cast*
T0*
_output_shapes
: 
X
training/Adam/add/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
a
training/Adam/addAddtraining/Adam/add/xtraining/Adam/mul*
T0*
_output_shapes
: 
\
training/Adam/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
m
training/Adam/truedivRealDivtraining/Adam/truediv/xtraining/Adam/add*
T0*
_output_shapes
: 
`
training/Adam/mul_1MulAdam/lr/readtraining/Adam/truediv*
T0*
_output_shapes
: 
r
training/Adam/Cast_1CastAdam/iterations/read*

SrcT0	*
Truncate( *

DstT0*
_output_shapes
: 
Z
training/Adam/add_1/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/add_1Addtraining/Adam/Cast_1training/Adam/add_1/y*
T0*
_output_shapes
: 
`
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add_1*
_output_shapes
: *
T0
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
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
_output_shapes
: *
T0
b
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add_1*
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
l
training/Adam/truediv_1RealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
i
training/Adam/mul_2Multraining/Adam/mul_1training/Adam/truediv_1*
T0*
_output_shapes
: 
|
#training/Adam/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"         @   *
dtype0
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
VariableV2*
	container *&
_output_shapes
:@*
shape:@*
shared_name *
dtype0
Ů
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*&
_output_shapes
:@

training/Adam/Variable/readIdentitytraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*&
_output_shapes
:@*
T0
b
training/Adam/zeros_1Const*
valueB@*    *
dtype0*
_output_shapes
:@
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
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
_output_shapes
:@*
T0*+
_class!
loc:@training/Adam/Variable_1
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
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*
T0*

index_type0*&
_output_shapes
:@@

training/Adam/Variable_2
VariableV2*
	container *&
_output_shapes
:@@*
shape:@@*
shared_name *
dtype0
á
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0
Ą
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*&
_output_shapes
:@@*
T0
b
training/Adam/zeros_3Const*
dtype0*
_output_shapes
:@*
valueB@*    

training/Adam/Variable_3
VariableV2*
	container *
_output_shapes
:@*
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
training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*'
_output_shapes
:@*
T0*

index_type0
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
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ö
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
_output_shapes	
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(

training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes	
:
~
%training/Adam/zeros_6/shape_as_tensorConst*
_output_shapes
:*%
valueB"            *
dtype0
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
dtype0*
	container *(
_output_shapes
:*
shape:
ă
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*(
_output_shapes
:
Ł
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*(
_output_shapes
:*
T0
d
training/Adam/zeros_7Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_7
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ö
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
_output_shapes	
:*
T0*+
_class!
loc:@training/Adam/Variable_7
~
%training/Adam/zeros_8/shape_as_tensorConst*%
valueB"            *
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
Ś
training/Adam/zeros_8Fill%training/Adam/zeros_8/shape_as_tensortraining/Adam/zeros_8/Const*(
_output_shapes
:*
T0*

index_type0
 
training/Adam/Variable_8
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:*
shape:
ă
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*(
_output_shapes
:
Ł
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
T0*+
_class!
loc:@training/Adam/Variable_8*(
_output_shapes
:
d
training/Adam/zeros_9Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_9
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ö
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes	
:*
use_locking(

training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes	
:

&training/Adam/zeros_10/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_10/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Š
training/Adam/zeros_10Fill&training/Adam/zeros_10/shape_as_tensortraining/Adam/zeros_10/Const*

index_type0*(
_output_shapes
:*
T0
Ą
training/Adam/Variable_10
VariableV2*
shape:*
shared_name *
dtype0*
	container *(
_output_shapes
:
ç
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*(
_output_shapes
:
e
training/Adam/zeros_11Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_11
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ú
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(

training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes	
:

&training/Adam/zeros_12/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_12/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_12Fill&training/Adam/zeros_12/shape_as_tensortraining/Adam/zeros_12/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_12
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
ç
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*(
_output_shapes
:
e
training/Adam/zeros_13Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_13
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
Ú
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_13

&training/Adam/zeros_14/shape_as_tensorConst*
_output_shapes
:*%
valueB"            *
dtype0
a
training/Adam/zeros_14/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Š
training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_14
VariableV2*
shape:*
shared_name *
dtype0*
	container *(
_output_shapes
:
ç
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*(
_output_shapes
:
e
training/Adam/zeros_15Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_15
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ú
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_15

&training/Adam/zeros_16/shape_as_tensorConst*%
valueB"            *
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
Š
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_16
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:*
shape:
ç
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16
Ś
training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_16
q
&training/Adam/zeros_17/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_17/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_17Fill&training/Adam/zeros_17/shape_as_tensortraining/Adam/zeros_17/Const*
T0*

index_type0*
_output_shapes	
:

training/Adam/Variable_17
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ú
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes	
:*
use_locking(

training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
_output_shapes	
:

&training/Adam/zeros_18/shape_as_tensorConst*%
valueB"            *
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
Š
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*

index_type0*(
_output_shapes
:*
T0
Ą
training/Adam/Variable_18
VariableV2*
shape:*
shared_name *
dtype0*
	container *(
_output_shapes
:
ç
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Ś
training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_18
q
&training/Adam/zeros_19/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_19/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_19Fill&training/Adam/zeros_19/shape_as_tensortraining/Adam/zeros_19/Const*
T0*

index_type0*
_output_shapes	
:

training/Adam/Variable_19
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ú
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
_output_shapes	
:

&training/Adam/zeros_20/shape_as_tensorConst*%
valueB"            *
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
Š
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_20
VariableV2*
shape:*
shared_name *
dtype0*
	container *(
_output_shapes
:
ç
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*(
_output_shapes
:*
T0
e
training/Adam/zeros_21Const*
_output_shapes	
:*
valueB*    *
dtype0

training/Adam/Variable_21
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ú
 training/Adam/Variable_21/AssignAssigntraining/Adam/Variable_21training/Adam/zeros_21*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes	
:*
use_locking(

training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*
T0*,
_class"
 loc:@training/Adam/Variable_21*
_output_shapes	
:

&training/Adam/zeros_22/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_22/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Š
training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_22
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
ç
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22
Ś
training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*
T0*,
_class"
 loc:@training/Adam/Variable_22*(
_output_shapes
:
e
training/Adam/zeros_23Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_23
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
Ú
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*
T0*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes	
:

&training/Adam/zeros_24/shape_as_tensorConst*%
valueB"            *
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
Š
training/Adam/zeros_24Fill&training/Adam/zeros_24/shape_as_tensortraining/Adam/zeros_24/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_24
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:*
shape:
ç
 training/Adam/Variable_24/AssignAssigntraining/Adam/Variable_24training/Adam/zeros_24*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24
Ś
training/Adam/Variable_24/readIdentitytraining/Adam/Variable_24*
T0*,
_class"
 loc:@training/Adam/Variable_24*(
_output_shapes
:
e
training/Adam/zeros_25Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_25
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ú
 training/Adam/Variable_25/AssignAssigntraining/Adam/Variable_25training/Adam/zeros_25*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25

training/Adam/Variable_25/readIdentitytraining/Adam/Variable_25*
T0*,
_class"
 loc:@training/Adam/Variable_25*
_output_shapes	
:

&training/Adam/zeros_26/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_26/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Š
training/Adam/zeros_26Fill&training/Adam/zeros_26/shape_as_tensortraining/Adam/zeros_26/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_26
VariableV2*
	container *(
_output_shapes
:*
shape:*
shared_name *
dtype0
ç
 training/Adam/Variable_26/AssignAssigntraining/Adam/Variable_26training/Adam/zeros_26*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_26*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_26/readIdentitytraining/Adam/Variable_26*
T0*,
_class"
 loc:@training/Adam/Variable_26*(
_output_shapes
:
e
training/Adam/zeros_27Const*
_output_shapes	
:*
valueB*    *
dtype0

training/Adam/Variable_27
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ú
 training/Adam/Variable_27/AssignAssigntraining/Adam/Variable_27training/Adam/zeros_27*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_27*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_27/readIdentitytraining/Adam/Variable_27*
T0*,
_class"
 loc:@training/Adam/Variable_27*
_output_shapes	
:

&training/Adam/zeros_28/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_28/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_28Fill&training/Adam/zeros_28/shape_as_tensortraining/Adam/zeros_28/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_28
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:*
shape:
ç
 training/Adam/Variable_28/AssignAssigntraining/Adam/Variable_28training/Adam/zeros_28*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_28*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_28/readIdentitytraining/Adam/Variable_28*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_28
e
training/Adam/zeros_29Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_29
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ú
 training/Adam/Variable_29/AssignAssigntraining/Adam/Variable_29training/Adam/zeros_29*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_29

training/Adam/Variable_29/readIdentitytraining/Adam/Variable_29*
T0*,
_class"
 loc:@training/Adam/Variable_29*
_output_shapes	
:

&training/Adam/zeros_30/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_30/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_30Fill&training/Adam/zeros_30/shape_as_tensortraining/Adam/zeros_30/Const*

index_type0*(
_output_shapes
:*
T0
Ą
training/Adam/Variable_30
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
ç
 training/Adam/Variable_30/AssignAssigntraining/Adam/Variable_30training/Adam/zeros_30*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_30
Ś
training/Adam/Variable_30/readIdentitytraining/Adam/Variable_30*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_30
e
training/Adam/zeros_31Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_31
VariableV2*
	container *
_output_shapes	
:*
shape:*
shared_name *
dtype0
Ú
 training/Adam/Variable_31/AssignAssigntraining/Adam/Variable_31training/Adam/zeros_31*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_31*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_31/readIdentitytraining/Adam/Variable_31*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_31

&training/Adam/zeros_32/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_32/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Š
training/Adam/zeros_32Fill&training/Adam/zeros_32/shape_as_tensortraining/Adam/zeros_32/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_32
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:*
shape:
ç
 training/Adam/Variable_32/AssignAssigntraining/Adam/Variable_32training/Adam/zeros_32*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_32*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_32/readIdentitytraining/Adam/Variable_32*
T0*,
_class"
 loc:@training/Adam/Variable_32*(
_output_shapes
:
e
training/Adam/zeros_33Const*
_output_shapes	
:*
valueB*    *
dtype0

training/Adam/Variable_33
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ú
 training/Adam/Variable_33/AssignAssigntraining/Adam/Variable_33training/Adam/zeros_33*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_33*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_33/readIdentitytraining/Adam/Variable_33*
T0*,
_class"
 loc:@training/Adam/Variable_33*
_output_shapes	
:

&training/Adam/zeros_34/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_34/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_34Fill&training/Adam/zeros_34/shape_as_tensortraining/Adam/zeros_34/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_34
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:*
shape:
ç
 training/Adam/Variable_34/AssignAssigntraining/Adam/Variable_34training/Adam/zeros_34*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_34*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_34/readIdentitytraining/Adam/Variable_34*
T0*,
_class"
 loc:@training/Adam/Variable_34*(
_output_shapes
:
e
training/Adam/zeros_35Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_35
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ú
 training/Adam/Variable_35/AssignAssigntraining/Adam/Variable_35training/Adam/zeros_35*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_35*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_35/readIdentitytraining/Adam/Variable_35*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_35

&training/Adam/zeros_36/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_36/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_36Fill&training/Adam/zeros_36/shape_as_tensortraining/Adam/zeros_36/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_36
VariableV2*
shape:*
shared_name *
dtype0*
	container *(
_output_shapes
:
ç
 training/Adam/Variable_36/AssignAssigntraining/Adam/Variable_36training/Adam/zeros_36*
T0*,
_class"
 loc:@training/Adam/Variable_36*
validate_shape(*(
_output_shapes
:*
use_locking(
Ś
training/Adam/Variable_36/readIdentitytraining/Adam/Variable_36*
T0*,
_class"
 loc:@training/Adam/Variable_36*(
_output_shapes
:
e
training/Adam/zeros_37Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_37
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ú
 training/Adam/Variable_37/AssignAssigntraining/Adam/Variable_37training/Adam/zeros_37*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_37*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_37/readIdentitytraining/Adam/Variable_37*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_37

&training/Adam/zeros_38/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_38/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¨
training/Adam/zeros_38Fill&training/Adam/zeros_38/shape_as_tensortraining/Adam/zeros_38/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_38
VariableV2*
shared_name *
dtype0*
	container *'
_output_shapes
:@*
shape:@
ć
 training/Adam/Variable_38/AssignAssigntraining/Adam/Variable_38training/Adam/zeros_38*,
_class"
 loc:@training/Adam/Variable_38*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0
Ľ
training/Adam/Variable_38/readIdentitytraining/Adam/Variable_38*'
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_38
c
training/Adam/zeros_39Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_39
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
Ů
 training/Adam/Variable_39/AssignAssigntraining/Adam/Variable_39training/Adam/zeros_39*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_39*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_39/readIdentitytraining/Adam/Variable_39*
T0*,
_class"
 loc:@training/Adam/Variable_39*
_output_shapes
:@

&training/Adam/zeros_40/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_40/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¨
training/Adam/zeros_40Fill&training/Adam/zeros_40/shape_as_tensortraining/Adam/zeros_40/Const*

index_type0*'
_output_shapes
:@*
T0

training/Adam/Variable_40
VariableV2*
shape:@*
shared_name *
dtype0*
	container *'
_output_shapes
:@
ć
 training/Adam/Variable_40/AssignAssigntraining/Adam/Variable_40training/Adam/zeros_40*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_40*
validate_shape(*'
_output_shapes
:@
Ľ
training/Adam/Variable_40/readIdentitytraining/Adam/Variable_40*,
_class"
 loc:@training/Adam/Variable_40*'
_output_shapes
:@*
T0
c
training/Adam/zeros_41Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_41
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
Ů
 training/Adam/Variable_41/AssignAssigntraining/Adam/Variable_41training/Adam/zeros_41*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_41*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_41/readIdentitytraining/Adam/Variable_41*
T0*,
_class"
 loc:@training/Adam/Variable_41*
_output_shapes
:@

&training/Adam/zeros_42/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_42/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
training/Adam/zeros_42Fill&training/Adam/zeros_42/shape_as_tensortraining/Adam/zeros_42/Const*&
_output_shapes
:@@*
T0*

index_type0

training/Adam/Variable_42
VariableV2*
shared_name *
dtype0*
	container *&
_output_shapes
:@@*
shape:@@
ĺ
 training/Adam/Variable_42/AssignAssigntraining/Adam/Variable_42training/Adam/zeros_42*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_42*
validate_shape(*&
_output_shapes
:@@
¤
training/Adam/Variable_42/readIdentitytraining/Adam/Variable_42*
T0*,
_class"
 loc:@training/Adam/Variable_42*&
_output_shapes
:@@
c
training/Adam/zeros_43Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_43
VariableV2*
	container *
_output_shapes
:@*
shape:@*
shared_name *
dtype0
Ů
 training/Adam/Variable_43/AssignAssigntraining/Adam/Variable_43training/Adam/zeros_43*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_43*
validate_shape(

training/Adam/Variable_43/readIdentitytraining/Adam/Variable_43*
T0*,
_class"
 loc:@training/Adam/Variable_43*
_output_shapes
:@
{
training/Adam/zeros_44Const*%
valueB@*    *
dtype0*&
_output_shapes
:@

training/Adam/Variable_44
VariableV2*
dtype0*
	container *&
_output_shapes
:@*
shape:@*
shared_name 
ĺ
 training/Adam/Variable_44/AssignAssigntraining/Adam/Variable_44training/Adam/zeros_44*,
_class"
 loc:@training/Adam/Variable_44*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
¤
training/Adam/Variable_44/readIdentitytraining/Adam/Variable_44*&
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_44
c
training/Adam/zeros_45Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_45
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
Ů
 training/Adam/Variable_45/AssignAssigntraining/Adam/Variable_45training/Adam/zeros_45*
T0*,
_class"
 loc:@training/Adam/Variable_45*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_45/readIdentitytraining/Adam/Variable_45*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_45

&training/Adam/zeros_46/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_46/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
§
training/Adam/zeros_46Fill&training/Adam/zeros_46/shape_as_tensortraining/Adam/zeros_46/Const*
T0*

index_type0*&
_output_shapes
:@

training/Adam/Variable_46
VariableV2*
	container *&
_output_shapes
:@*
shape:@*
shared_name *
dtype0
ĺ
 training/Adam/Variable_46/AssignAssigntraining/Adam/Variable_46training/Adam/zeros_46*,
_class"
 loc:@training/Adam/Variable_46*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
¤
training/Adam/Variable_46/readIdentitytraining/Adam/Variable_46*&
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_46
c
training/Adam/zeros_47Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_47
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
Ů
 training/Adam/Variable_47/AssignAssigntraining/Adam/Variable_47training/Adam/zeros_47*
T0*,
_class"
 loc:@training/Adam/Variable_47*
validate_shape(*
_output_shapes
:@*
use_locking(

training/Adam/Variable_47/readIdentitytraining/Adam/Variable_47*
T0*,
_class"
 loc:@training/Adam/Variable_47*
_output_shapes
:@

&training/Adam/zeros_48/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_48/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
training/Adam/zeros_48Fill&training/Adam/zeros_48/shape_as_tensortraining/Adam/zeros_48/Const*
T0*

index_type0*&
_output_shapes
:@@

training/Adam/Variable_48
VariableV2*
dtype0*
	container *&
_output_shapes
:@@*
shape:@@*
shared_name 
ĺ
 training/Adam/Variable_48/AssignAssigntraining/Adam/Variable_48training/Adam/zeros_48*,
_class"
 loc:@training/Adam/Variable_48*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0
¤
training/Adam/Variable_48/readIdentitytraining/Adam/Variable_48*
T0*,
_class"
 loc:@training/Adam/Variable_48*&
_output_shapes
:@@
c
training/Adam/zeros_49Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_49
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:@*
shape:@
Ů
 training/Adam/Variable_49/AssignAssigntraining/Adam/Variable_49training/Adam/zeros_49*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_49*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_49/readIdentitytraining/Adam/Variable_49*
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_49

&training/Adam/zeros_50/shape_as_tensorConst*%
valueB"      @      *
dtype0*
_output_shapes
:
a
training/Adam/zeros_50/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
¨
training/Adam/zeros_50Fill&training/Adam/zeros_50/shape_as_tensortraining/Adam/zeros_50/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_50
VariableV2*
dtype0*
	container *'
_output_shapes
:@*
shape:@*
shared_name 
ć
 training/Adam/Variable_50/AssignAssigntraining/Adam/Variable_50training/Adam/zeros_50*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_50*
validate_shape(*'
_output_shapes
:@
Ľ
training/Adam/Variable_50/readIdentitytraining/Adam/Variable_50*'
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_50
e
training/Adam/zeros_51Const*
_output_shapes	
:*
valueB*    *
dtype0

training/Adam/Variable_51
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ú
 training/Adam/Variable_51/AssignAssigntraining/Adam/Variable_51training/Adam/zeros_51*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_51*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_51/readIdentitytraining/Adam/Variable_51*
T0*,
_class"
 loc:@training/Adam/Variable_51*
_output_shapes	
:

&training/Adam/zeros_52/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_52/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_52Fill&training/Adam/zeros_52/shape_as_tensortraining/Adam/zeros_52/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_52
VariableV2*
shape:*
shared_name *
dtype0*
	container *(
_output_shapes
:
ç
 training/Adam/Variable_52/AssignAssigntraining/Adam/Variable_52training/Adam/zeros_52*,
_class"
 loc:@training/Adam/Variable_52*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Ś
training/Adam/Variable_52/readIdentitytraining/Adam/Variable_52*
T0*,
_class"
 loc:@training/Adam/Variable_52*(
_output_shapes
:
e
training/Adam/zeros_53Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_53
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
Ú
 training/Adam/Variable_53/AssignAssigntraining/Adam/Variable_53training/Adam/zeros_53*,
_class"
 loc:@training/Adam/Variable_53*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

training/Adam/Variable_53/readIdentitytraining/Adam/Variable_53*
T0*,
_class"
 loc:@training/Adam/Variable_53*
_output_shapes	
:

&training/Adam/zeros_54/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_54/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Š
training/Adam/zeros_54Fill&training/Adam/zeros_54/shape_as_tensortraining/Adam/zeros_54/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_54
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
ç
 training/Adam/Variable_54/AssignAssigntraining/Adam/Variable_54training/Adam/zeros_54*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_54*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_54/readIdentitytraining/Adam/Variable_54*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_54
e
training/Adam/zeros_55Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_55
VariableV2*
	container *
_output_shapes	
:*
shape:*
shared_name *
dtype0
Ú
 training/Adam/Variable_55/AssignAssigntraining/Adam/Variable_55training/Adam/zeros_55*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_55

training/Adam/Variable_55/readIdentitytraining/Adam/Variable_55*
T0*,
_class"
 loc:@training/Adam/Variable_55*
_output_shapes	
:

&training/Adam/zeros_56/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_56/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_56Fill&training/Adam/zeros_56/shape_as_tensortraining/Adam/zeros_56/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_56
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:*
shape:
ç
 training/Adam/Variable_56/AssignAssigntraining/Adam/Variable_56training/Adam/zeros_56*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_56*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_56/readIdentitytraining/Adam/Variable_56*
T0*,
_class"
 loc:@training/Adam/Variable_56*(
_output_shapes
:
e
training/Adam/zeros_57Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_57
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ú
 training/Adam/Variable_57/AssignAssigntraining/Adam/Variable_57training/Adam/zeros_57*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_57*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_57/readIdentitytraining/Adam/Variable_57*
T0*,
_class"
 loc:@training/Adam/Variable_57*
_output_shapes	
:

&training/Adam/zeros_58/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"            
a
training/Adam/zeros_58/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_58Fill&training/Adam/zeros_58/shape_as_tensortraining/Adam/zeros_58/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_58
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
ç
 training/Adam/Variable_58/AssignAssigntraining/Adam/Variable_58training/Adam/zeros_58*,
_class"
 loc:@training/Adam/Variable_58*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Ś
training/Adam/Variable_58/readIdentitytraining/Adam/Variable_58*
T0*,
_class"
 loc:@training/Adam/Variable_58*(
_output_shapes
:
e
training/Adam/zeros_59Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_59
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ú
 training/Adam/Variable_59/AssignAssigntraining/Adam/Variable_59training/Adam/zeros_59*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_59*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_59/readIdentitytraining/Adam/Variable_59*
T0*,
_class"
 loc:@training/Adam/Variable_59*
_output_shapes	
:

&training/Adam/zeros_60/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"            
a
training/Adam/zeros_60/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_60Fill&training/Adam/zeros_60/shape_as_tensortraining/Adam/zeros_60/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_60
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:*
shape:
ç
 training/Adam/Variable_60/AssignAssigntraining/Adam/Variable_60training/Adam/zeros_60*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_60*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_60/readIdentitytraining/Adam/Variable_60*
T0*,
_class"
 loc:@training/Adam/Variable_60*(
_output_shapes
:
e
training/Adam/zeros_61Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_61
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ú
 training/Adam/Variable_61/AssignAssigntraining/Adam/Variable_61training/Adam/zeros_61*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_61*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_61/readIdentitytraining/Adam/Variable_61*
T0*,
_class"
 loc:@training/Adam/Variable_61*
_output_shapes	
:

&training/Adam/zeros_62/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_62/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Š
training/Adam/zeros_62Fill&training/Adam/zeros_62/shape_as_tensortraining/Adam/zeros_62/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_62
VariableV2*
shape:*
shared_name *
dtype0*
	container *(
_output_shapes
:
ç
 training/Adam/Variable_62/AssignAssigntraining/Adam/Variable_62training/Adam/zeros_62*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_62*
validate_shape(
Ś
training/Adam/Variable_62/readIdentitytraining/Adam/Variable_62*
T0*,
_class"
 loc:@training/Adam/Variable_62*(
_output_shapes
:
q
&training/Adam/zeros_63/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_63/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_63Fill&training/Adam/zeros_63/shape_as_tensortraining/Adam/zeros_63/Const*
T0*

index_type0*
_output_shapes	
:

training/Adam/Variable_63
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ú
 training/Adam/Variable_63/AssignAssigntraining/Adam/Variable_63training/Adam/zeros_63*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_63*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_63/readIdentitytraining/Adam/Variable_63*
T0*,
_class"
 loc:@training/Adam/Variable_63*
_output_shapes	
:

&training/Adam/zeros_64/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_64/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Š
training/Adam/zeros_64Fill&training/Adam/zeros_64/shape_as_tensortraining/Adam/zeros_64/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_64
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
ç
 training/Adam/Variable_64/AssignAssigntraining/Adam/Variable_64training/Adam/zeros_64*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_64*
validate_shape(
Ś
training/Adam/Variable_64/readIdentitytraining/Adam/Variable_64*
T0*,
_class"
 loc:@training/Adam/Variable_64*(
_output_shapes
:
q
&training/Adam/zeros_65/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_65/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_65Fill&training/Adam/zeros_65/shape_as_tensortraining/Adam/zeros_65/Const*
T0*

index_type0*
_output_shapes	
:

training/Adam/Variable_65
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
Ú
 training/Adam/Variable_65/AssignAssigntraining/Adam/Variable_65training/Adam/zeros_65*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_65*
validate_shape(

training/Adam/Variable_65/readIdentitytraining/Adam/Variable_65*
T0*,
_class"
 loc:@training/Adam/Variable_65*
_output_shapes	
:

&training/Adam/zeros_66/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_66/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_66Fill&training/Adam/zeros_66/shape_as_tensortraining/Adam/zeros_66/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_66
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
ç
 training/Adam/Variable_66/AssignAssigntraining/Adam/Variable_66training/Adam/zeros_66*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_66
Ś
training/Adam/Variable_66/readIdentitytraining/Adam/Variable_66*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_66
e
training/Adam/zeros_67Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_67
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
Ú
 training/Adam/Variable_67/AssignAssigntraining/Adam/Variable_67training/Adam/zeros_67*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_67*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_67/readIdentitytraining/Adam/Variable_67*
T0*,
_class"
 loc:@training/Adam/Variable_67*
_output_shapes	
:

&training/Adam/zeros_68/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_68/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_68Fill&training/Adam/zeros_68/shape_as_tensortraining/Adam/zeros_68/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_68
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:*
shape:
ç
 training/Adam/Variable_68/AssignAssigntraining/Adam/Variable_68training/Adam/zeros_68*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_68
Ś
training/Adam/Variable_68/readIdentitytraining/Adam/Variable_68*
T0*,
_class"
 loc:@training/Adam/Variable_68*(
_output_shapes
:
e
training/Adam/zeros_69Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_69
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
Ú
 training/Adam/Variable_69/AssignAssigntraining/Adam/Variable_69training/Adam/zeros_69*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_69*
validate_shape(

training/Adam/Variable_69/readIdentitytraining/Adam/Variable_69*,
_class"
 loc:@training/Adam/Variable_69*
_output_shapes	
:*
T0

&training/Adam/zeros_70/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_70/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_70Fill&training/Adam/zeros_70/shape_as_tensortraining/Adam/zeros_70/Const*

index_type0*(
_output_shapes
:*
T0
Ą
training/Adam/Variable_70
VariableV2*
	container *(
_output_shapes
:*
shape:*
shared_name *
dtype0
ç
 training/Adam/Variable_70/AssignAssigntraining/Adam/Variable_70training/Adam/zeros_70*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_70*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_70/readIdentitytraining/Adam/Variable_70*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_70
e
training/Adam/zeros_71Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_71
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ú
 training/Adam/Variable_71/AssignAssigntraining/Adam/Variable_71training/Adam/zeros_71*,
_class"
 loc:@training/Adam/Variable_71*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

training/Adam/Variable_71/readIdentitytraining/Adam/Variable_71*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_71

&training/Adam/zeros_72/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_72/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Š
training/Adam/zeros_72Fill&training/Adam/zeros_72/shape_as_tensortraining/Adam/zeros_72/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_72
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
ç
 training/Adam/Variable_72/AssignAssigntraining/Adam/Variable_72training/Adam/zeros_72*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_72*
validate_shape(
Ś
training/Adam/Variable_72/readIdentitytraining/Adam/Variable_72*
T0*,
_class"
 loc:@training/Adam/Variable_72*(
_output_shapes
:
e
training/Adam/zeros_73Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_73
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
Ú
 training/Adam/Variable_73/AssignAssigntraining/Adam/Variable_73training/Adam/zeros_73*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_73*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_73/readIdentitytraining/Adam/Variable_73*
_output_shapes	
:*
T0*,
_class"
 loc:@training/Adam/Variable_73

&training/Adam/zeros_74/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_74/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_74Fill&training/Adam/zeros_74/shape_as_tensortraining/Adam/zeros_74/Const*
T0*

index_type0*(
_output_shapes
:
Ą
training/Adam/Variable_74
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
ç
 training/Adam/Variable_74/AssignAssigntraining/Adam/Variable_74training/Adam/zeros_74*
T0*,
_class"
 loc:@training/Adam/Variable_74*
validate_shape(*(
_output_shapes
:*
use_locking(
Ś
training/Adam/Variable_74/readIdentitytraining/Adam/Variable_74*
T0*,
_class"
 loc:@training/Adam/Variable_74*(
_output_shapes
:
e
training/Adam/zeros_75Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_75
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ú
 training/Adam/Variable_75/AssignAssigntraining/Adam/Variable_75training/Adam/zeros_75*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_75*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_75/readIdentitytraining/Adam/Variable_75*
T0*,
_class"
 loc:@training/Adam/Variable_75*
_output_shapes	
:

&training/Adam/zeros_76/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_76/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Š
training/Adam/zeros_76Fill&training/Adam/zeros_76/shape_as_tensortraining/Adam/zeros_76/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_76
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
ç
 training/Adam/Variable_76/AssignAssigntraining/Adam/Variable_76training/Adam/zeros_76*,
_class"
 loc:@training/Adam/Variable_76*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Ś
training/Adam/Variable_76/readIdentitytraining/Adam/Variable_76*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_76
e
training/Adam/zeros_77Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_77
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
Ú
 training/Adam/Variable_77/AssignAssigntraining/Adam/Variable_77training/Adam/zeros_77*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_77*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_77/readIdentitytraining/Adam/Variable_77*
T0*,
_class"
 loc:@training/Adam/Variable_77*
_output_shapes	
:

&training/Adam/zeros_78/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"            
a
training/Adam/zeros_78/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_78Fill&training/Adam/zeros_78/shape_as_tensortraining/Adam/zeros_78/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_78
VariableV2*
shape:*
shared_name *
dtype0*
	container *(
_output_shapes
:
ç
 training/Adam/Variable_78/AssignAssigntraining/Adam/Variable_78training/Adam/zeros_78*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_78*
validate_shape(*(
_output_shapes
:
Ś
training/Adam/Variable_78/readIdentitytraining/Adam/Variable_78*(
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_78
e
training/Adam/zeros_79Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_79
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ú
 training/Adam/Variable_79/AssignAssigntraining/Adam/Variable_79training/Adam/zeros_79*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_79

training/Adam/Variable_79/readIdentitytraining/Adam/Variable_79*
T0*,
_class"
 loc:@training/Adam/Variable_79*
_output_shapes	
:

&training/Adam/zeros_80/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_80/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
training/Adam/zeros_80Fill&training/Adam/zeros_80/shape_as_tensortraining/Adam/zeros_80/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_80
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
ç
 training/Adam/Variable_80/AssignAssigntraining/Adam/Variable_80training/Adam/zeros_80*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_80
Ś
training/Adam/Variable_80/readIdentitytraining/Adam/Variable_80*,
_class"
 loc:@training/Adam/Variable_80*(
_output_shapes
:*
T0
e
training/Adam/zeros_81Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_81
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:
Ú
 training/Adam/Variable_81/AssignAssigntraining/Adam/Variable_81training/Adam/zeros_81*
T0*,
_class"
 loc:@training/Adam/Variable_81*
validate_shape(*
_output_shapes	
:*
use_locking(

training/Adam/Variable_81/readIdentitytraining/Adam/Variable_81*
T0*,
_class"
 loc:@training/Adam/Variable_81*
_output_shapes	
:

&training/Adam/zeros_82/shape_as_tensorConst*%
valueB"            *
dtype0*
_output_shapes
:
a
training/Adam/zeros_82/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Š
training/Adam/zeros_82Fill&training/Adam/zeros_82/shape_as_tensortraining/Adam/zeros_82/Const*(
_output_shapes
:*
T0*

index_type0
Ą
training/Adam/Variable_82
VariableV2*
dtype0*
	container *(
_output_shapes
:*
shape:*
shared_name 
ç
 training/Adam/Variable_82/AssignAssigntraining/Adam/Variable_82training/Adam/zeros_82*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_82*
validate_shape(
Ś
training/Adam/Variable_82/readIdentitytraining/Adam/Variable_82*
T0*,
_class"
 loc:@training/Adam/Variable_82*(
_output_shapes
:
e
training/Adam/zeros_83Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_83
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
Ú
 training/Adam/Variable_83/AssignAssigntraining/Adam/Variable_83training/Adam/zeros_83*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_83*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_83/readIdentitytraining/Adam/Variable_83*,
_class"
 loc:@training/Adam/Variable_83*
_output_shapes	
:*
T0

&training/Adam/zeros_84/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_84/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
¨
training/Adam/zeros_84Fill&training/Adam/zeros_84/shape_as_tensortraining/Adam/zeros_84/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_84
VariableV2*
shared_name *
dtype0*
	container *'
_output_shapes
:@*
shape:@
ć
 training/Adam/Variable_84/AssignAssigntraining/Adam/Variable_84training/Adam/zeros_84*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_84*
validate_shape(*'
_output_shapes
:@
Ľ
training/Adam/Variable_84/readIdentitytraining/Adam/Variable_84*,
_class"
 loc:@training/Adam/Variable_84*'
_output_shapes
:@*
T0
c
training/Adam/zeros_85Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_85
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
Ů
 training/Adam/Variable_85/AssignAssigntraining/Adam/Variable_85training/Adam/zeros_85*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_85*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_85/readIdentitytraining/Adam/Variable_85*
T0*,
_class"
 loc:@training/Adam/Variable_85*
_output_shapes
:@

&training/Adam/zeros_86/shape_as_tensorConst*%
valueB"         @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_86/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¨
training/Adam/zeros_86Fill&training/Adam/zeros_86/shape_as_tensortraining/Adam/zeros_86/Const*
T0*

index_type0*'
_output_shapes
:@

training/Adam/Variable_86
VariableV2*
shared_name *
dtype0*
	container *'
_output_shapes
:@*
shape:@
ć
 training/Adam/Variable_86/AssignAssigntraining/Adam/Variable_86training/Adam/zeros_86*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_86*
validate_shape(*'
_output_shapes
:@
Ľ
training/Adam/Variable_86/readIdentitytraining/Adam/Variable_86*'
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_86
c
training/Adam/zeros_87Const*
_output_shapes
:@*
valueB@*    *
dtype0

training/Adam/Variable_87
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
Ů
 training/Adam/Variable_87/AssignAssigntraining/Adam/Variable_87training/Adam/zeros_87*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_87*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_87/readIdentitytraining/Adam/Variable_87*
T0*,
_class"
 loc:@training/Adam/Variable_87*
_output_shapes
:@

&training/Adam/zeros_88/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_88/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
training/Adam/zeros_88Fill&training/Adam/zeros_88/shape_as_tensortraining/Adam/zeros_88/Const*

index_type0*&
_output_shapes
:@@*
T0

training/Adam/Variable_88
VariableV2*
shape:@@*
shared_name *
dtype0*
	container *&
_output_shapes
:@@
ĺ
 training/Adam/Variable_88/AssignAssigntraining/Adam/Variable_88training/Adam/zeros_88*
T0*,
_class"
 loc:@training/Adam/Variable_88*
validate_shape(*&
_output_shapes
:@@*
use_locking(
¤
training/Adam/Variable_88/readIdentitytraining/Adam/Variable_88*,
_class"
 loc:@training/Adam/Variable_88*&
_output_shapes
:@@*
T0
c
training/Adam/zeros_89Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/Adam/Variable_89
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
Ů
 training/Adam/Variable_89/AssignAssigntraining/Adam/Variable_89training/Adam/zeros_89*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_89*
validate_shape(*
_output_shapes
:@

training/Adam/Variable_89/readIdentitytraining/Adam/Variable_89*
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_89
{
training/Adam/zeros_90Const*%
valueB@*    *
dtype0*&
_output_shapes
:@

training/Adam/Variable_90
VariableV2*
shape:@*
shared_name *
dtype0*
	container *&
_output_shapes
:@
ĺ
 training/Adam/Variable_90/AssignAssigntraining/Adam/Variable_90training/Adam/zeros_90*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_90
¤
training/Adam/Variable_90/readIdentitytraining/Adam/Variable_90*
T0*,
_class"
 loc:@training/Adam/Variable_90*&
_output_shapes
:@
c
training/Adam/zeros_91Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_91
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
Ů
 training/Adam/Variable_91/AssignAssigntraining/Adam/Variable_91training/Adam/zeros_91*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_91*
validate_shape(*
_output_shapes
:

training/Adam/Variable_91/readIdentitytraining/Adam/Variable_91*,
_class"
 loc:@training/Adam/Variable_91*
_output_shapes
:*
T0
p
&training/Adam/zeros_92/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_92/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_92Fill&training/Adam/zeros_92/shape_as_tensortraining/Adam/zeros_92/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_92
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
Ů
 training/Adam/Variable_92/AssignAssigntraining/Adam/Variable_92training/Adam/zeros_92*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_92*
validate_shape(*
_output_shapes
:

training/Adam/Variable_92/readIdentitytraining/Adam/Variable_92*,
_class"
 loc:@training/Adam/Variable_92*
_output_shapes
:*
T0
p
&training/Adam/zeros_93/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_93/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_93Fill&training/Adam/zeros_93/shape_as_tensortraining/Adam/zeros_93/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_93
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
Ů
 training/Adam/Variable_93/AssignAssigntraining/Adam/Variable_93training/Adam/zeros_93*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_93*
validate_shape(*
_output_shapes
:

training/Adam/Variable_93/readIdentitytraining/Adam/Variable_93*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_93
p
&training/Adam/zeros_94/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_94/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_94Fill&training/Adam/zeros_94/shape_as_tensortraining/Adam/zeros_94/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_94
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ů
 training/Adam/Variable_94/AssignAssigntraining/Adam/Variable_94training/Adam/zeros_94*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_94*
validate_shape(*
_output_shapes
:

training/Adam/Variable_94/readIdentitytraining/Adam/Variable_94*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_94
p
&training/Adam/zeros_95/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_95/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_95Fill&training/Adam/zeros_95/shape_as_tensortraining/Adam/zeros_95/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_95
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ů
 training/Adam/Variable_95/AssignAssigntraining/Adam/Variable_95training/Adam/zeros_95*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_95*
validate_shape(

training/Adam/Variable_95/readIdentitytraining/Adam/Variable_95*
T0*,
_class"
 loc:@training/Adam/Variable_95*
_output_shapes
:
p
&training/Adam/zeros_96/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_96/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_96Fill&training/Adam/zeros_96/shape_as_tensortraining/Adam/zeros_96/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_96
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ů
 training/Adam/Variable_96/AssignAssigntraining/Adam/Variable_96training/Adam/zeros_96*,
_class"
 loc:@training/Adam/Variable_96*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_96/readIdentitytraining/Adam/Variable_96*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_96
p
&training/Adam/zeros_97/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_97/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_97Fill&training/Adam/zeros_97/shape_as_tensortraining/Adam/zeros_97/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_97
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ů
 training/Adam/Variable_97/AssignAssigntraining/Adam/Variable_97training/Adam/zeros_97*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_97*
validate_shape(*
_output_shapes
:

training/Adam/Variable_97/readIdentitytraining/Adam/Variable_97*
T0*,
_class"
 loc:@training/Adam/Variable_97*
_output_shapes
:
p
&training/Adam/zeros_98/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_98/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_98Fill&training/Adam/zeros_98/shape_as_tensortraining/Adam/zeros_98/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_98
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ů
 training/Adam/Variable_98/AssignAssigntraining/Adam/Variable_98training/Adam/zeros_98*,
_class"
 loc:@training/Adam/Variable_98*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_98/readIdentitytraining/Adam/Variable_98*,
_class"
 loc:@training/Adam/Variable_98*
_output_shapes
:*
T0
p
&training/Adam/zeros_99/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_99/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_99Fill&training/Adam/zeros_99/shape_as_tensortraining/Adam/zeros_99/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_99
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
Ů
 training/Adam/Variable_99/AssignAssigntraining/Adam/Variable_99training/Adam/zeros_99*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_99*
validate_shape(*
_output_shapes
:

training/Adam/Variable_99/readIdentitytraining/Adam/Variable_99*
T0*,
_class"
 loc:@training/Adam/Variable_99*
_output_shapes
:
q
'training/Adam/zeros_100/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_100/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_100Fill'training/Adam/zeros_100/shape_as_tensortraining/Adam/zeros_100/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_100
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ý
!training/Adam/Variable_100/AssignAssigntraining/Adam/Variable_100training/Adam/zeros_100*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_100

training/Adam/Variable_100/readIdentitytraining/Adam/Variable_100*
T0*-
_class#
!loc:@training/Adam/Variable_100*
_output_shapes
:
q
'training/Adam/zeros_101/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_101/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_101Fill'training/Adam/zeros_101/shape_as_tensortraining/Adam/zeros_101/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_101
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
Ý
!training/Adam/Variable_101/AssignAssigntraining/Adam/Variable_101training/Adam/zeros_101*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_101*
validate_shape(

training/Adam/Variable_101/readIdentitytraining/Adam/Variable_101*
T0*-
_class#
!loc:@training/Adam/Variable_101*
_output_shapes
:
q
'training/Adam/zeros_102/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
b
training/Adam/zeros_102/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_102Fill'training/Adam/zeros_102/shape_as_tensortraining/Adam/zeros_102/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_102
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
Ý
!training/Adam/Variable_102/AssignAssigntraining/Adam/Variable_102training/Adam/zeros_102*-
_class#
!loc:@training/Adam/Variable_102*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_102/readIdentitytraining/Adam/Variable_102*
T0*-
_class#
!loc:@training/Adam/Variable_102*
_output_shapes
:
q
'training/Adam/zeros_103/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_103/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_103Fill'training/Adam/zeros_103/shape_as_tensortraining/Adam/zeros_103/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_103
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ý
!training/Adam/Variable_103/AssignAssigntraining/Adam/Variable_103training/Adam/zeros_103*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_103*
validate_shape(*
_output_shapes
:

training/Adam/Variable_103/readIdentitytraining/Adam/Variable_103*
T0*-
_class#
!loc:@training/Adam/Variable_103*
_output_shapes
:
q
'training/Adam/zeros_104/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_104/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_104Fill'training/Adam/zeros_104/shape_as_tensortraining/Adam/zeros_104/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_104
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
Ý
!training/Adam/Variable_104/AssignAssigntraining/Adam/Variable_104training/Adam/zeros_104*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_104*
validate_shape(*
_output_shapes
:

training/Adam/Variable_104/readIdentitytraining/Adam/Variable_104*
T0*-
_class#
!loc:@training/Adam/Variable_104*
_output_shapes
:
q
'training/Adam/zeros_105/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_105/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_105Fill'training/Adam/zeros_105/shape_as_tensortraining/Adam/zeros_105/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_105
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
Ý
!training/Adam/Variable_105/AssignAssigntraining/Adam/Variable_105training/Adam/zeros_105*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_105*
validate_shape(*
_output_shapes
:

training/Adam/Variable_105/readIdentitytraining/Adam/Variable_105*
T0*-
_class#
!loc:@training/Adam/Variable_105*
_output_shapes
:
q
'training/Adam/zeros_106/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_106/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_106Fill'training/Adam/zeros_106/shape_as_tensortraining/Adam/zeros_106/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_106
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ý
!training/Adam/Variable_106/AssignAssigntraining/Adam/Variable_106training/Adam/zeros_106*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_106*
validate_shape(

training/Adam/Variable_106/readIdentitytraining/Adam/Variable_106*
T0*-
_class#
!loc:@training/Adam/Variable_106*
_output_shapes
:
q
'training/Adam/zeros_107/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
b
training/Adam/zeros_107/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_107Fill'training/Adam/zeros_107/shape_as_tensortraining/Adam/zeros_107/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_107
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ý
!training/Adam/Variable_107/AssignAssigntraining/Adam/Variable_107training/Adam/zeros_107*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_107*
validate_shape(*
_output_shapes
:

training/Adam/Variable_107/readIdentitytraining/Adam/Variable_107*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_107
q
'training/Adam/zeros_108/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_108/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_108Fill'training/Adam/zeros_108/shape_as_tensortraining/Adam/zeros_108/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_108
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
Ý
!training/Adam/Variable_108/AssignAssigntraining/Adam/Variable_108training/Adam/zeros_108*-
_class#
!loc:@training/Adam/Variable_108*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_108/readIdentitytraining/Adam/Variable_108*
T0*-
_class#
!loc:@training/Adam/Variable_108*
_output_shapes
:
q
'training/Adam/zeros_109/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
b
training/Adam/zeros_109/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_109Fill'training/Adam/zeros_109/shape_as_tensortraining/Adam/zeros_109/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_109
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ý
!training/Adam/Variable_109/AssignAssigntraining/Adam/Variable_109training/Adam/zeros_109*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_109*
validate_shape(*
_output_shapes
:

training/Adam/Variable_109/readIdentitytraining/Adam/Variable_109*-
_class#
!loc:@training/Adam/Variable_109*
_output_shapes
:*
T0
q
'training/Adam/zeros_110/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
b
training/Adam/zeros_110/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_110Fill'training/Adam/zeros_110/shape_as_tensortraining/Adam/zeros_110/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_110
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
Ý
!training/Adam/Variable_110/AssignAssigntraining/Adam/Variable_110training/Adam/zeros_110*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_110*
validate_shape(*
_output_shapes
:

training/Adam/Variable_110/readIdentitytraining/Adam/Variable_110*
T0*-
_class#
!loc:@training/Adam/Variable_110*
_output_shapes
:
q
'training/Adam/zeros_111/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
b
training/Adam/zeros_111/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_111Fill'training/Adam/zeros_111/shape_as_tensortraining/Adam/zeros_111/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_111
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
Ý
!training/Adam/Variable_111/AssignAssigntraining/Adam/Variable_111training/Adam/zeros_111*
T0*-
_class#
!loc:@training/Adam/Variable_111*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_111/readIdentitytraining/Adam/Variable_111*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_111
q
'training/Adam/zeros_112/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_112/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_112Fill'training/Adam/zeros_112/shape_as_tensortraining/Adam/zeros_112/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_112
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ý
!training/Adam/Variable_112/AssignAssigntraining/Adam/Variable_112training/Adam/zeros_112*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_112*
validate_shape(*
_output_shapes
:

training/Adam/Variable_112/readIdentitytraining/Adam/Variable_112*
T0*-
_class#
!loc:@training/Adam/Variable_112*
_output_shapes
:
q
'training/Adam/zeros_113/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_113/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_113Fill'training/Adam/zeros_113/shape_as_tensortraining/Adam/zeros_113/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_113
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
Ý
!training/Adam/Variable_113/AssignAssigntraining/Adam/Variable_113training/Adam/zeros_113*-
_class#
!loc:@training/Adam/Variable_113*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_113/readIdentitytraining/Adam/Variable_113*-
_class#
!loc:@training/Adam/Variable_113*
_output_shapes
:*
T0
q
'training/Adam/zeros_114/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_114/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_114Fill'training/Adam/zeros_114/shape_as_tensortraining/Adam/zeros_114/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_114
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ý
!training/Adam/Variable_114/AssignAssigntraining/Adam/Variable_114training/Adam/zeros_114*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_114*
validate_shape(*
_output_shapes
:

training/Adam/Variable_114/readIdentitytraining/Adam/Variable_114*
T0*-
_class#
!loc:@training/Adam/Variable_114*
_output_shapes
:
q
'training/Adam/zeros_115/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_115/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_115Fill'training/Adam/zeros_115/shape_as_tensortraining/Adam/zeros_115/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_115
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
Ý
!training/Adam/Variable_115/AssignAssigntraining/Adam/Variable_115training/Adam/zeros_115*
T0*-
_class#
!loc:@training/Adam/Variable_115*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_115/readIdentitytraining/Adam/Variable_115*
T0*-
_class#
!loc:@training/Adam/Variable_115*
_output_shapes
:
q
'training/Adam/zeros_116/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_116/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_116Fill'training/Adam/zeros_116/shape_as_tensortraining/Adam/zeros_116/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_116
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
Ý
!training/Adam/Variable_116/AssignAssigntraining/Adam/Variable_116training/Adam/zeros_116*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_116

training/Adam/Variable_116/readIdentitytraining/Adam/Variable_116*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_116
q
'training/Adam/zeros_117/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_117/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_117Fill'training/Adam/zeros_117/shape_as_tensortraining/Adam/zeros_117/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_117
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ý
!training/Adam/Variable_117/AssignAssigntraining/Adam/Variable_117training/Adam/zeros_117*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_117*
validate_shape(

training/Adam/Variable_117/readIdentitytraining/Adam/Variable_117*
T0*-
_class#
!loc:@training/Adam/Variable_117*
_output_shapes
:
q
'training/Adam/zeros_118/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_118/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

training/Adam/zeros_118Fill'training/Adam/zeros_118/shape_as_tensortraining/Adam/zeros_118/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_118
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
Ý
!training/Adam/Variable_118/AssignAssigntraining/Adam/Variable_118training/Adam/zeros_118*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_118*
validate_shape(*
_output_shapes
:

training/Adam/Variable_118/readIdentitytraining/Adam/Variable_118*
T0*-
_class#
!loc:@training/Adam/Variable_118*
_output_shapes
:
q
'training/Adam/zeros_119/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_119/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_119Fill'training/Adam/zeros_119/shape_as_tensortraining/Adam/zeros_119/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_119
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
Ý
!training/Adam/Variable_119/AssignAssigntraining/Adam/Variable_119training/Adam/zeros_119*
T0*-
_class#
!loc:@training/Adam/Variable_119*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_119/readIdentitytraining/Adam/Variable_119*
T0*-
_class#
!loc:@training/Adam/Variable_119*
_output_shapes
:
q
'training/Adam/zeros_120/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_120/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_120Fill'training/Adam/zeros_120/shape_as_tensortraining/Adam/zeros_120/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_120
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
Ý
!training/Adam/Variable_120/AssignAssigntraining/Adam/Variable_120training/Adam/zeros_120*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_120*
validate_shape(

training/Adam/Variable_120/readIdentitytraining/Adam/Variable_120*
T0*-
_class#
!loc:@training/Adam/Variable_120*
_output_shapes
:
q
'training/Adam/zeros_121/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_121/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_121Fill'training/Adam/zeros_121/shape_as_tensortraining/Adam/zeros_121/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_121
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
Ý
!training/Adam/Variable_121/AssignAssigntraining/Adam/Variable_121training/Adam/zeros_121*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_121*
validate_shape(*
_output_shapes
:

training/Adam/Variable_121/readIdentitytraining/Adam/Variable_121*-
_class#
!loc:@training/Adam/Variable_121*
_output_shapes
:*
T0
q
'training/Adam/zeros_122/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_122/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_122Fill'training/Adam/zeros_122/shape_as_tensortraining/Adam/zeros_122/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_122
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
Ý
!training/Adam/Variable_122/AssignAssigntraining/Adam/Variable_122training/Adam/zeros_122*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_122*
validate_shape(*
_output_shapes
:

training/Adam/Variable_122/readIdentitytraining/Adam/Variable_122*
T0*-
_class#
!loc:@training/Adam/Variable_122*
_output_shapes
:
q
'training/Adam/zeros_123/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_123/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_123Fill'training/Adam/zeros_123/shape_as_tensortraining/Adam/zeros_123/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_123
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ý
!training/Adam/Variable_123/AssignAssigntraining/Adam/Variable_123training/Adam/zeros_123*-
_class#
!loc:@training/Adam/Variable_123*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_123/readIdentitytraining/Adam/Variable_123*
T0*-
_class#
!loc:@training/Adam/Variable_123*
_output_shapes
:
q
'training/Adam/zeros_124/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_124/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_124Fill'training/Adam/zeros_124/shape_as_tensortraining/Adam/zeros_124/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_124
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ý
!training/Adam/Variable_124/AssignAssigntraining/Adam/Variable_124training/Adam/zeros_124*-
_class#
!loc:@training/Adam/Variable_124*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_124/readIdentitytraining/Adam/Variable_124*
T0*-
_class#
!loc:@training/Adam/Variable_124*
_output_shapes
:
q
'training/Adam/zeros_125/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
b
training/Adam/zeros_125/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_125Fill'training/Adam/zeros_125/shape_as_tensortraining/Adam/zeros_125/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_125
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
Ý
!training/Adam/Variable_125/AssignAssigntraining/Adam/Variable_125training/Adam/zeros_125*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_125*
validate_shape(*
_output_shapes
:

training/Adam/Variable_125/readIdentitytraining/Adam/Variable_125*
T0*-
_class#
!loc:@training/Adam/Variable_125*
_output_shapes
:
q
'training/Adam/zeros_126/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_126/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_126Fill'training/Adam/zeros_126/shape_as_tensortraining/Adam/zeros_126/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_126
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ý
!training/Adam/Variable_126/AssignAssigntraining/Adam/Variable_126training/Adam/zeros_126*
T0*-
_class#
!loc:@training/Adam/Variable_126*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_126/readIdentitytraining/Adam/Variable_126*
T0*-
_class#
!loc:@training/Adam/Variable_126*
_output_shapes
:
q
'training/Adam/zeros_127/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_127/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_127Fill'training/Adam/zeros_127/shape_as_tensortraining/Adam/zeros_127/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_127
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ý
!training/Adam/Variable_127/AssignAssigntraining/Adam/Variable_127training/Adam/zeros_127*-
_class#
!loc:@training/Adam/Variable_127*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_127/readIdentitytraining/Adam/Variable_127*
T0*-
_class#
!loc:@training/Adam/Variable_127*
_output_shapes
:
q
'training/Adam/zeros_128/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_128/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_128Fill'training/Adam/zeros_128/shape_as_tensortraining/Adam/zeros_128/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_128
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
Ý
!training/Adam/Variable_128/AssignAssigntraining/Adam/Variable_128training/Adam/zeros_128*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_128*
validate_shape(

training/Adam/Variable_128/readIdentitytraining/Adam/Variable_128*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_128
q
'training/Adam/zeros_129/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_129/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_129Fill'training/Adam/zeros_129/shape_as_tensortraining/Adam/zeros_129/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_129
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
Ý
!training/Adam/Variable_129/AssignAssigntraining/Adam/Variable_129training/Adam/zeros_129*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_129*
validate_shape(*
_output_shapes
:

training/Adam/Variable_129/readIdentitytraining/Adam/Variable_129*
T0*-
_class#
!loc:@training/Adam/Variable_129*
_output_shapes
:
q
'training/Adam/zeros_130/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_130/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_130Fill'training/Adam/zeros_130/shape_as_tensortraining/Adam/zeros_130/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_130
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
Ý
!training/Adam/Variable_130/AssignAssigntraining/Adam/Variable_130training/Adam/zeros_130*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_130*
validate_shape(

training/Adam/Variable_130/readIdentitytraining/Adam/Variable_130*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_130
q
'training/Adam/zeros_131/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_131/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_131Fill'training/Adam/zeros_131/shape_as_tensortraining/Adam/zeros_131/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_131
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ý
!training/Adam/Variable_131/AssignAssigntraining/Adam/Variable_131training/Adam/zeros_131*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_131*
validate_shape(*
_output_shapes
:

training/Adam/Variable_131/readIdentitytraining/Adam/Variable_131*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_131
q
'training/Adam/zeros_132/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_132/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_132Fill'training/Adam/zeros_132/shape_as_tensortraining/Adam/zeros_132/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_132
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
Ý
!training/Adam/Variable_132/AssignAssigntraining/Adam/Variable_132training/Adam/zeros_132*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_132*
validate_shape(*
_output_shapes
:

training/Adam/Variable_132/readIdentitytraining/Adam/Variable_132*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_132
q
'training/Adam/zeros_133/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_133/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_133Fill'training/Adam/zeros_133/shape_as_tensortraining/Adam/zeros_133/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_133
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
Ý
!training/Adam/Variable_133/AssignAssigntraining/Adam/Variable_133training/Adam/zeros_133*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_133*
validate_shape(*
_output_shapes
:

training/Adam/Variable_133/readIdentitytraining/Adam/Variable_133*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_133
q
'training/Adam/zeros_134/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_134/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_134Fill'training/Adam/zeros_134/shape_as_tensortraining/Adam/zeros_134/Const*

index_type0*
_output_shapes
:*
T0

training/Adam/Variable_134
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
Ý
!training/Adam/Variable_134/AssignAssigntraining/Adam/Variable_134training/Adam/zeros_134*-
_class#
!loc:@training/Adam/Variable_134*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

training/Adam/Variable_134/readIdentitytraining/Adam/Variable_134*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_134
q
'training/Adam/zeros_135/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_135/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_135Fill'training/Adam/zeros_135/shape_as_tensortraining/Adam/zeros_135/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_135
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Ý
!training/Adam/Variable_135/AssignAssigntraining/Adam/Variable_135training/Adam/zeros_135*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_135*
validate_shape(*
_output_shapes
:

training/Adam/Variable_135/readIdentitytraining/Adam/Variable_135*
T0*-
_class#
!loc:@training/Adam/Variable_135*
_output_shapes
:
q
'training/Adam/zeros_136/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_136/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_136Fill'training/Adam/zeros_136/shape_as_tensortraining/Adam/zeros_136/Const*
_output_shapes
:*
T0*

index_type0

training/Adam/Variable_136
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
Ý
!training/Adam/Variable_136/AssignAssigntraining/Adam/Variable_136training/Adam/zeros_136*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_136*
validate_shape(*
_output_shapes
:

training/Adam/Variable_136/readIdentitytraining/Adam/Variable_136*
T0*-
_class#
!loc:@training/Adam/Variable_136*
_output_shapes
:
q
'training/Adam/zeros_137/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training/Adam/zeros_137/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

training/Adam/zeros_137Fill'training/Adam/zeros_137/shape_as_tensortraining/Adam/zeros_137/Const*
T0*

index_type0*
_output_shapes
:

training/Adam/Variable_137
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
Ý
!training/Adam/Variable_137/AssignAssigntraining/Adam/Variable_137training/Adam/zeros_137*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@training/Adam/Variable_137

training/Adam/Variable_137/readIdentitytraining/Adam/Variable_137*
_output_shapes
:*
T0*-
_class#
!loc:@training/Adam/Variable_137
z
training/Adam/mul_3MulAdam/beta_1/readtraining/Adam/Variable/read*&
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
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ś
training/Adam/mul_4Multraining/Adam/sub_2Dtraining/Adam/gradients/conv1a/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
u
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*&
_output_shapes
:@
}
training/Adam/mul_5MulAdam/beta_2/readtraining/Adam/Variable_46/read*
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
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/SquareSquareDtraining/Adam/gradients/conv1a/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
v
training/Adam/mul_6Multraining/Adam/sub_3training/Adam/Square*
T0*&
_output_shapes
:@
u
training/Adam/add_3Addtraining/Adam/mul_5training/Adam/mul_6*&
_output_shapes
:@*
T0
u
training/Adam/mul_7Multraining/Adam/mul_2training/Adam/add_2*
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
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_3training/Adam/Const_3*&
_output_shapes
:@*
T0

training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*&
_output_shapes
:@*
T0
l
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*&
_output_shapes
:@
Z
training/Adam/add_4/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
x
training/Adam/add_4Addtraining/Adam/Sqrt_1training/Adam/add_4/y*&
_output_shapes
:@*
T0
}
training/Adam/truediv_2RealDivtraining/Adam/mul_7training/Adam/add_4*
T0*&
_output_shapes
:@
x
training/Adam/sub_4Subconv1a/kernel/readtraining/Adam/truediv_2*
T0*&
_output_shapes
:@
Đ
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_2*)
_class
loc:@training/Adam/Variable*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
Ř
training/Adam/Assign_1Assigntraining/Adam/Variable_46training/Adam/add_3*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_46*
validate_shape(*&
_output_shapes
:@
Ŕ
training/Adam/Assign_2Assignconv1a/kerneltraining/Adam/sub_4* 
_class
loc:@conv1a/kernel*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
p
training/Adam/mul_8MulAdam/beta_1/readtraining/Adam/Variable_1/read*
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
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_9Multraining/Adam/sub_57training/Adam/gradients/conv1a/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
:@
r
training/Adam/mul_10MulAdam/beta_2/readtraining/Adam/Variable_47/read*
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
~
training/Adam/Square_1Square7training/Adam/gradients/conv1a/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
m
training/Adam/mul_11Multraining/Adam/sub_6training/Adam/Square_1*
_output_shapes
:@*
T0
k
training/Adam/add_6Addtraining/Adam/mul_10training/Adam/mul_11*
_output_shapes
:@*
T0
j
training/Adam/mul_12Multraining/Adam/mul_2training/Adam/add_5*
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
training/Adam/Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *  

%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_6training/Adam/Const_5*
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
training/Adam/add_7/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
l
training/Adam/add_7Addtraining/Adam/Sqrt_2training/Adam/add_7/y*
_output_shapes
:@*
T0
r
training/Adam/truediv_3RealDivtraining/Adam/mul_12training/Adam/add_7*
T0*
_output_shapes
:@
j
training/Adam/sub_7Subconv1a/bias/readtraining/Adam/truediv_3*
_output_shapes
:@*
T0
Ę
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_5*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
Ě
training/Adam/Assign_4Assigntraining/Adam/Variable_47training/Adam/add_6*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_47*
validate_shape(*
_output_shapes
:@
°
training/Adam/Assign_5Assignconv1a/biastraining/Adam/sub_7*
use_locking(*
T0*
_class
loc:@conv1a/bias*
validate_shape(*
_output_shapes
:@
}
training/Adam/mul_13MulAdam/beta_1/readtraining/Adam/Variable_2/read*
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
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
_output_shapes
: *
T0
§
training/Adam/mul_14Multraining/Adam/sub_8Dtraining/Adam/gradients/conv1b/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@@
w
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*&
_output_shapes
:@@*
T0
~
training/Adam/mul_15MulAdam/beta_2/readtraining/Adam/Variable_48/read*
T0*&
_output_shapes
:@@
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

training/Adam/Square_2SquareDtraining/Adam/gradients/conv1b/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
y
training/Adam/mul_16Multraining/Adam/sub_9training/Adam/Square_2*
T0*&
_output_shapes
:@@
w
training/Adam/add_9Addtraining/Adam/mul_15training/Adam/mul_16*
T0*&
_output_shapes
:@@
v
training/Adam/mul_17Multraining/Adam/mul_2training/Adam/add_8*&
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
training/Adam/Const_7Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_9training/Adam/Const_7*
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
[
training/Adam/add_10/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
z
training/Adam/add_10Addtraining/Adam/Sqrt_3training/Adam/add_10/y*
T0*&
_output_shapes
:@@

training/Adam/truediv_4RealDivtraining/Adam/mul_17training/Adam/add_10*&
_output_shapes
:@@*
T0
y
training/Adam/sub_10Subconv1b/kernel/readtraining/Adam/truediv_4*
T0*&
_output_shapes
:@@
Ö
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_8*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*&
_output_shapes
:@@
Ř
training/Adam/Assign_7Assigntraining/Adam/Variable_48training/Adam/add_9*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_48*
validate_shape(*&
_output_shapes
:@@
Á
training/Adam/Assign_8Assignconv1b/kerneltraining/Adam/sub_10*&
_output_shapes
:@@*
use_locking(*
T0* 
_class
loc:@conv1b/kernel*
validate_shape(
q
training/Adam/mul_18MulAdam/beta_1/readtraining/Adam/Variable_3/read*
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

training/Adam/mul_19Multraining/Adam/sub_117training/Adam/gradients/conv1b/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
:@
r
training/Adam/mul_20MulAdam/beta_2/readtraining/Adam/Variable_49/read*
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
~
training/Adam/Square_3Square7training/Adam/gradients/conv1b/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
n
training/Adam/mul_21Multraining/Adam/sub_12training/Adam/Square_3*
_output_shapes
:@*
T0
l
training/Adam/add_12Addtraining/Adam/mul_20training/Adam/mul_21*
T0*
_output_shapes
:@
k
training/Adam/mul_22Multraining/Adam/mul_2training/Adam/add_11*
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
training/Adam/Const_9Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_12training/Adam/Const_9*
_output_shapes
:@*
T0
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
training/Adam/add_13/yConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
n
training/Adam/add_13Addtraining/Adam/Sqrt_4training/Adam/add_13/y*
_output_shapes
:@*
T0
s
training/Adam/truediv_5RealDivtraining/Adam/mul_22training/Adam/add_13*
_output_shapes
:@*
T0
k
training/Adam/sub_13Subconv1b/bias/readtraining/Adam/truediv_5*
_output_shapes
:@*
T0
Ë
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_11*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:@
Î
training/Adam/Assign_10Assigntraining/Adam/Variable_49training/Adam/add_12*
T0*,
_class"
 loc:@training/Adam/Variable_49*
validate_shape(*
_output_shapes
:@*
use_locking(
˛
training/Adam/Assign_11Assignconv1b/biastraining/Adam/sub_13*
use_locking(*
T0*
_class
loc:@conv1b/bias*
validate_shape(*
_output_shapes
:@
~
training/Adam/mul_23MulAdam/beta_1/readtraining/Adam/Variable_4/read*'
_output_shapes
:@*
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
Š
training/Adam/mul_24Multraining/Adam/sub_14Dtraining/Adam/gradients/conv2a/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
y
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*'
_output_shapes
:@*
T0

training/Adam/mul_25MulAdam/beta_2/readtraining/Adam/Variable_50/read*
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

training/Adam/Square_4SquareDtraining/Adam/gradients/conv2a/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
{
training/Adam/mul_26Multraining/Adam/sub_15training/Adam/Square_4*'
_output_shapes
:@*
T0
y
training/Adam/add_15Addtraining/Adam/mul_25training/Adam/mul_26*
T0*'
_output_shapes
:@
x
training/Adam/mul_27Multraining/Adam/mul_2training/Adam/add_14*
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
%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_15training/Adam/Const_11*'
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
training/Adam/add_16/yConst*
_output_shapes
: *
valueB
 *żÖ3*
dtype0
{
training/Adam/add_16Addtraining/Adam/Sqrt_5training/Adam/add_16/y*
T0*'
_output_shapes
:@

training/Adam/truediv_6RealDivtraining/Adam/mul_27training/Adam/add_16*'
_output_shapes
:@*
T0
z
training/Adam/sub_16Subconv2a/kernel/readtraining/Adam/truediv_6*
T0*'
_output_shapes
:@
Ů
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_14*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*'
_output_shapes
:@
Ű
training/Adam/Assign_13Assigntraining/Adam/Variable_50training/Adam/add_15*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_50
Ă
training/Adam/Assign_14Assignconv2a/kerneltraining/Adam/sub_16*
use_locking(*
T0* 
_class
loc:@conv2a/kernel*
validate_shape(*'
_output_shapes
:@
r
training/Adam/mul_28MulAdam/beta_1/readtraining/Adam/Variable_5/read*
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
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_29Multraining/Adam/sub_177training/Adam/gradients/conv2a/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
m
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes	
:
s
training/Adam/mul_30MulAdam/beta_2/readtraining/Adam/Variable_51/read*
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

training/Adam/Square_5Square7training/Adam/gradients/conv2a/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/mul_31Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes	
:
m
training/Adam/add_18Addtraining/Adam/mul_30training/Adam/mul_31*
_output_shapes	
:*
T0
l
training/Adam/mul_32Multraining/Adam/mul_2training/Adam/add_17*
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
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_18training/Adam/Const_13*
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
training/Adam/add_19/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
o
training/Adam/add_19Addtraining/Adam/Sqrt_6training/Adam/add_19/y*
_output_shapes	
:*
T0
t
training/Adam/truediv_7RealDivtraining/Adam/mul_32training/Adam/add_19*
T0*
_output_shapes	
:
l
training/Adam/sub_19Subconv2a/bias/readtraining/Adam/truediv_7*
_output_shapes	
:*
T0
Í
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_17*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_16Assigntraining/Adam/Variable_51training/Adam/add_18*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_51*
validate_shape(*
_output_shapes	
:
ł
training/Adam/Assign_17Assignconv2a/biastraining/Adam/sub_19*
use_locking(*
T0*
_class
loc:@conv2a/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_33MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*(
_output_shapes
:
[
training/Adam/sub_20/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ş
training/Adam/mul_34Multraining/Adam/sub_20Dtraining/Adam/gradients/conv2b/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
z
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*(
_output_shapes
:*
T0

training/Adam/mul_35MulAdam/beta_2/readtraining/Adam/Variable_52/read*
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

training/Adam/Square_6SquareDtraining/Adam/gradients/conv2b/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
|
training/Adam/mul_36Multraining/Adam/sub_21training/Adam/Square_6*
T0*(
_output_shapes
:
z
training/Adam/add_21Addtraining/Adam/mul_35training/Adam/mul_36*(
_output_shapes
:*
T0
y
training/Adam/mul_37Multraining/Adam/mul_2training/Adam/add_20*(
_output_shapes
:*
T0
[
training/Adam/Const_14Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_15Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_21training/Adam/Const_15*
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
training/Adam/add_22/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
|
training/Adam/add_22Addtraining/Adam/Sqrt_7training/Adam/add_22/y*
T0*(
_output_shapes
:

training/Adam/truediv_8RealDivtraining/Adam/mul_37training/Adam/add_22*(
_output_shapes
:*
T0
{
training/Adam/sub_22Subconv2b/kernel/readtraining/Adam/truediv_8*
T0*(
_output_shapes
:
Ú
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_20*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_19Assigntraining/Adam/Variable_52training/Adam/add_21*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_52*
validate_shape(*(
_output_shapes
:
Ä
training/Adam/Assign_20Assignconv2b/kerneltraining/Adam/sub_22*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv2b/kernel
r
training/Adam/mul_38MulAdam/beta_1/readtraining/Adam/Variable_7/read*
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
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_39Multraining/Adam/sub_237training/Adam/gradients/conv2b/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes	
:
s
training/Adam/mul_40MulAdam/beta_2/readtraining/Adam/Variable_53/read*
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

training/Adam/Square_7Square7training/Adam/gradients/conv2b/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/mul_41Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes	
:
m
training/Adam/add_24Addtraining/Adam/mul_40training/Adam/mul_41*
T0*
_output_shapes	
:
l
training/Adam/mul_42Multraining/Adam/mul_2training/Adam/add_23*
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
%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_24training/Adam/Const_17*
T0*
_output_shapes	
:

training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_16*
_output_shapes	
:*
T0
a
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
_output_shapes	
:*
T0
[
training/Adam/add_25/yConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
o
training/Adam/add_25Addtraining/Adam/Sqrt_8training/Adam/add_25/y*
_output_shapes	
:*
T0
t
training/Adam/truediv_9RealDivtraining/Adam/mul_42training/Adam/add_25*
T0*
_output_shapes	
:
l
training/Adam/sub_25Subconv2b/bias/readtraining/Adam/truediv_9*
_output_shapes	
:*
T0
Í
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_23*
_output_shapes	
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(
Ď
training/Adam/Assign_22Assigntraining/Adam/Variable_53training/Adam/add_24*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_53*
validate_shape(
ł
training/Adam/Assign_23Assignconv2b/biastraining/Adam/sub_25*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv2b/bias

training/Adam/mul_43MulAdam/beta_1/readtraining/Adam/Variable_8/read*
T0*(
_output_shapes
:
[
training/Adam/sub_26/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_26Subtraining/Adam/sub_26/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ş
training/Adam/mul_44Multraining/Adam/sub_26Dtraining/Adam/gradients/conv3a/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
z
training/Adam/add_26Addtraining/Adam/mul_43training/Adam/mul_44*
T0*(
_output_shapes
:

training/Adam/mul_45MulAdam/beta_2/readtraining/Adam/Variable_54/read*
T0*(
_output_shapes
:
[
training/Adam/sub_27/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_27Subtraining/Adam/sub_27/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_8SquareDtraining/Adam/gradients/conv3a/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
|
training/Adam/mul_46Multraining/Adam/sub_27training/Adam/Square_8*
T0*(
_output_shapes
:
z
training/Adam/add_27Addtraining/Adam/mul_45training/Adam/mul_46*(
_output_shapes
:*
T0
y
training/Adam/mul_47Multraining/Adam/mul_2training/Adam/add_26*(
_output_shapes
:*
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

%training/Adam/clip_by_value_9/MinimumMinimumtraining/Adam/add_27training/Adam/Const_19*
T0*(
_output_shapes
:

training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_18*(
_output_shapes
:*
T0
n
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
T0*(
_output_shapes
:
[
training/Adam/add_28/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
|
training/Adam/add_28Addtraining/Adam/Sqrt_9training/Adam/add_28/y*
T0*(
_output_shapes
:

training/Adam/truediv_10RealDivtraining/Adam/mul_47training/Adam/add_28*
T0*(
_output_shapes
:
|
training/Adam/sub_28Subconv3a/kernel/readtraining/Adam/truediv_10*
T0*(
_output_shapes
:
Ú
training/Adam/Assign_24Assigntraining/Adam/Variable_8training/Adam/add_26*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_25Assigntraining/Adam/Variable_54training/Adam/add_27*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_54*
validate_shape(
Ä
training/Adam/Assign_26Assignconv3a/kerneltraining/Adam/sub_28*
T0* 
_class
loc:@conv3a/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
r
training/Adam/mul_48MulAdam/beta_1/readtraining/Adam/Variable_9/read*
_output_shapes	
:*
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

training/Adam/mul_49Multraining/Adam/sub_297training/Adam/gradients/conv3a/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_29Addtraining/Adam/mul_48training/Adam/mul_49*
T0*
_output_shapes	
:
s
training/Adam/mul_50MulAdam/beta_2/readtraining/Adam/Variable_55/read*
_output_shapes	
:*
T0
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

training/Adam/Square_9Square7training/Adam/gradients/conv3a/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/mul_51Multraining/Adam/sub_30training/Adam/Square_9*
T0*
_output_shapes	
:
m
training/Adam/add_30Addtraining/Adam/mul_50training/Adam/mul_51*
T0*
_output_shapes	
:
l
training/Adam/mul_52Multraining/Adam/mul_2training/Adam/add_29*
_output_shapes	
:*
T0
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

&training/Adam/clip_by_value_10/MinimumMinimumtraining/Adam/add_30training/Adam/Const_21*
_output_shapes	
:*
T0

training/Adam/clip_by_value_10Maximum&training/Adam/clip_by_value_10/Minimumtraining/Adam/Const_20*
_output_shapes	
:*
T0
c
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
_output_shapes	
:*
T0
[
training/Adam/add_31/yConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
p
training/Adam/add_31Addtraining/Adam/Sqrt_10training/Adam/add_31/y*
T0*
_output_shapes	
:
u
training/Adam/truediv_11RealDivtraining/Adam/mul_52training/Adam/add_31*
_output_shapes	
:*
T0
m
training/Adam/sub_31Subconv3a/bias/readtraining/Adam/truediv_11*
T0*
_output_shapes	
:
Í
training/Adam/Assign_27Assigntraining/Adam/Variable_9training/Adam/add_29*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes	
:*
use_locking(
Ď
training/Adam/Assign_28Assigntraining/Adam/Variable_55training/Adam/add_30*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_55*
validate_shape(*
_output_shapes	
:
ł
training/Adam/Assign_29Assignconv3a/biastraining/Adam/sub_31*
use_locking(*
T0*
_class
loc:@conv3a/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_53MulAdam/beta_1/readtraining/Adam/Variable_10/read*(
_output_shapes
:*
T0
[
training/Adam/sub_32/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_32Subtraining/Adam/sub_32/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ş
training/Adam/mul_54Multraining/Adam/sub_32Dtraining/Adam/gradients/conv3b/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
z
training/Adam/add_32Addtraining/Adam/mul_53training/Adam/mul_54*
T0*(
_output_shapes
:

training/Adam/mul_55MulAdam/beta_2/readtraining/Adam/Variable_56/read*
T0*(
_output_shapes
:
[
training/Adam/sub_33/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_33Subtraining/Adam/sub_33/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_10SquareDtraining/Adam/gradients/conv3b/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
}
training/Adam/mul_56Multraining/Adam/sub_33training/Adam/Square_10*(
_output_shapes
:*
T0
z
training/Adam/add_33Addtraining/Adam/mul_55training/Adam/mul_56*
T0*(
_output_shapes
:
y
training/Adam/mul_57Multraining/Adam/mul_2training/Adam/add_32*(
_output_shapes
:*
T0
[
training/Adam/Const_22Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_23Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_11/MinimumMinimumtraining/Adam/add_33training/Adam/Const_23*
T0*(
_output_shapes
:

training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_22*(
_output_shapes
:*
T0
p
training/Adam/Sqrt_11Sqrttraining/Adam/clip_by_value_11*
T0*(
_output_shapes
:
[
training/Adam/add_34/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_34Addtraining/Adam/Sqrt_11training/Adam/add_34/y*
T0*(
_output_shapes
:

training/Adam/truediv_12RealDivtraining/Adam/mul_57training/Adam/add_34*(
_output_shapes
:*
T0
|
training/Adam/sub_34Subconv3b/kernel/readtraining/Adam/truediv_12*
T0*(
_output_shapes
:
Ü
training/Adam/Assign_30Assigntraining/Adam/Variable_10training/Adam/add_32*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_31Assigntraining/Adam/Variable_56training/Adam/add_33*,
_class"
 loc:@training/Adam/Variable_56*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Ä
training/Adam/Assign_32Assignconv3b/kerneltraining/Adam/sub_34*
T0* 
_class
loc:@conv3b/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
s
training/Adam/mul_58MulAdam/beta_1/readtraining/Adam/Variable_11/read*
T0*
_output_shapes	
:
[
training/Adam/sub_35/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_35Subtraining/Adam/sub_35/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_59Multraining/Adam/sub_357training/Adam/gradients/conv3b/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_35Addtraining/Adam/mul_58training/Adam/mul_59*
_output_shapes	
:*
T0
s
training/Adam/mul_60MulAdam/beta_2/readtraining/Adam/Variable_57/read*
_output_shapes	
:*
T0
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

training/Adam/Square_11Square7training/Adam/gradients/conv3b/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
p
training/Adam/mul_61Multraining/Adam/sub_36training/Adam/Square_11*
T0*
_output_shapes	
:
m
training/Adam/add_36Addtraining/Adam/mul_60training/Adam/mul_61*
T0*
_output_shapes	
:
l
training/Adam/mul_62Multraining/Adam/mul_2training/Adam/add_35*
_output_shapes	
:*
T0
[
training/Adam/Const_24Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_25Const*
_output_shapes
: *
valueB
 *  *
dtype0

&training/Adam/clip_by_value_12/MinimumMinimumtraining/Adam/add_36training/Adam/Const_25*
T0*
_output_shapes	
:

training/Adam/clip_by_value_12Maximum&training/Adam/clip_by_value_12/Minimumtraining/Adam/Const_24*
_output_shapes	
:*
T0
c
training/Adam/Sqrt_12Sqrttraining/Adam/clip_by_value_12*
_output_shapes	
:*
T0
[
training/Adam/add_37/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_37Addtraining/Adam/Sqrt_12training/Adam/add_37/y*
T0*
_output_shapes	
:
u
training/Adam/truediv_13RealDivtraining/Adam/mul_62training/Adam/add_37*
_output_shapes	
:*
T0
m
training/Adam/sub_37Subconv3b/bias/readtraining/Adam/truediv_13*
T0*
_output_shapes	
:
Ď
training/Adam/Assign_33Assigntraining/Adam/Variable_11training/Adam/add_35*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_34Assigntraining/Adam/Variable_57training/Adam/add_36*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_57*
validate_shape(*
_output_shapes	
:
ł
training/Adam/Assign_35Assignconv3b/biastraining/Adam/sub_37*
use_locking(*
T0*
_class
loc:@conv3b/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_63MulAdam/beta_1/readtraining/Adam/Variable_12/read*(
_output_shapes
:*
T0
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
training/Adam/mul_64Multraining/Adam/sub_38Dtraining/Adam/gradients/conv4a/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
z
training/Adam/add_38Addtraining/Adam/mul_63training/Adam/mul_64*(
_output_shapes
:*
T0

training/Adam/mul_65MulAdam/beta_2/readtraining/Adam/Variable_58/read*
T0*(
_output_shapes
:
[
training/Adam/sub_39/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_39Subtraining/Adam/sub_39/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_12SquareDtraining/Adam/gradients/conv4a/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
}
training/Adam/mul_66Multraining/Adam/sub_39training/Adam/Square_12*
T0*(
_output_shapes
:
z
training/Adam/add_39Addtraining/Adam/mul_65training/Adam/mul_66*(
_output_shapes
:*
T0
y
training/Adam/mul_67Multraining/Adam/mul_2training/Adam/add_38*(
_output_shapes
:*
T0
[
training/Adam/Const_26Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_27Const*
dtype0*
_output_shapes
: *
valueB
 *  

&training/Adam/clip_by_value_13/MinimumMinimumtraining/Adam/add_39training/Adam/Const_27*(
_output_shapes
:*
T0

training/Adam/clip_by_value_13Maximum&training/Adam/clip_by_value_13/Minimumtraining/Adam/Const_26*
T0*(
_output_shapes
:
p
training/Adam/Sqrt_13Sqrttraining/Adam/clip_by_value_13*
T0*(
_output_shapes
:
[
training/Adam/add_40/yConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
}
training/Adam/add_40Addtraining/Adam/Sqrt_13training/Adam/add_40/y*
T0*(
_output_shapes
:

training/Adam/truediv_14RealDivtraining/Adam/mul_67training/Adam/add_40*
T0*(
_output_shapes
:
|
training/Adam/sub_40Subconv4a/kernel/readtraining/Adam/truediv_14*(
_output_shapes
:*
T0
Ü
training/Adam/Assign_36Assigntraining/Adam/Variable_12training/Adam/add_38*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_37Assigntraining/Adam/Variable_58training/Adam/add_39*
T0*,
_class"
 loc:@training/Adam/Variable_58*
validate_shape(*(
_output_shapes
:*
use_locking(
Ä
training/Adam/Assign_38Assignconv4a/kerneltraining/Adam/sub_40*
use_locking(*
T0* 
_class
loc:@conv4a/kernel*
validate_shape(*(
_output_shapes
:
s
training/Adam/mul_68MulAdam/beta_1/readtraining/Adam/Variable_13/read*
_output_shapes	
:*
T0
[
training/Adam/sub_41/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_41Subtraining/Adam/sub_41/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_69Multraining/Adam/sub_417training/Adam/gradients/conv4a/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_41Addtraining/Adam/mul_68training/Adam/mul_69*
_output_shapes	
:*
T0
s
training/Adam/mul_70MulAdam/beta_2/readtraining/Adam/Variable_59/read*
T0*
_output_shapes	
:
[
training/Adam/sub_42/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_42Subtraining/Adam/sub_42/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_13Square7training/Adam/gradients/conv4a/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
p
training/Adam/mul_71Multraining/Adam/sub_42training/Adam/Square_13*
T0*
_output_shapes	
:
m
training/Adam/add_42Addtraining/Adam/mul_70training/Adam/mul_71*
_output_shapes	
:*
T0
l
training/Adam/mul_72Multraining/Adam/mul_2training/Adam/add_41*
_output_shapes	
:*
T0
[
training/Adam/Const_28Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_29Const*
_output_shapes
: *
valueB
 *  *
dtype0

&training/Adam/clip_by_value_14/MinimumMinimumtraining/Adam/add_42training/Adam/Const_29*
T0*
_output_shapes	
:

training/Adam/clip_by_value_14Maximum&training/Adam/clip_by_value_14/Minimumtraining/Adam/Const_28*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_14Sqrttraining/Adam/clip_by_value_14*
_output_shapes	
:*
T0
[
training/Adam/add_43/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_43Addtraining/Adam/Sqrt_14training/Adam/add_43/y*
T0*
_output_shapes	
:
u
training/Adam/truediv_15RealDivtraining/Adam/mul_72training/Adam/add_43*
_output_shapes	
:*
T0
m
training/Adam/sub_43Subconv4a/bias/readtraining/Adam/truediv_15*
T0*
_output_shapes	
:
Ď
training/Adam/Assign_39Assigntraining/Adam/Variable_13training/Adam/add_41*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_40Assigntraining/Adam/Variable_59training/Adam/add_42*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_59*
validate_shape(*
_output_shapes	
:
ł
training/Adam/Assign_41Assignconv4a/biastraining/Adam/sub_43*
use_locking(*
T0*
_class
loc:@conv4a/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_73MulAdam/beta_1/readtraining/Adam/Variable_14/read*
T0*(
_output_shapes
:
[
training/Adam/sub_44/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_44Subtraining/Adam/sub_44/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ş
training/Adam/mul_74Multraining/Adam/sub_44Dtraining/Adam/gradients/conv4b/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
z
training/Adam/add_44Addtraining/Adam/mul_73training/Adam/mul_74*(
_output_shapes
:*
T0

training/Adam/mul_75MulAdam/beta_2/readtraining/Adam/Variable_60/read*
T0*(
_output_shapes
:
[
training/Adam/sub_45/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_45Subtraining/Adam/sub_45/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_14SquareDtraining/Adam/gradients/conv4b/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
}
training/Adam/mul_76Multraining/Adam/sub_45training/Adam/Square_14*(
_output_shapes
:*
T0
z
training/Adam/add_45Addtraining/Adam/mul_75training/Adam/mul_76*
T0*(
_output_shapes
:
y
training/Adam/mul_77Multraining/Adam/mul_2training/Adam/add_44*(
_output_shapes
:*
T0
[
training/Adam/Const_30Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_31Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_15/MinimumMinimumtraining/Adam/add_45training/Adam/Const_31*(
_output_shapes
:*
T0

training/Adam/clip_by_value_15Maximum&training/Adam/clip_by_value_15/Minimumtraining/Adam/Const_30*
T0*(
_output_shapes
:
p
training/Adam/Sqrt_15Sqrttraining/Adam/clip_by_value_15*
T0*(
_output_shapes
:
[
training/Adam/add_46/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_46Addtraining/Adam/Sqrt_15training/Adam/add_46/y*
T0*(
_output_shapes
:

training/Adam/truediv_16RealDivtraining/Adam/mul_77training/Adam/add_46*
T0*(
_output_shapes
:
|
training/Adam/sub_46Subconv4b/kernel/readtraining/Adam/truediv_16*(
_output_shapes
:*
T0
Ü
training/Adam/Assign_42Assigntraining/Adam/Variable_14training/Adam/add_44*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_43Assigntraining/Adam/Variable_60training/Adam/add_45*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_60*
validate_shape(*(
_output_shapes
:
Ä
training/Adam/Assign_44Assignconv4b/kerneltraining/Adam/sub_46*(
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv4b/kernel*
validate_shape(
s
training/Adam/mul_78MulAdam/beta_1/readtraining/Adam/Variable_15/read*
T0*
_output_shapes	
:
[
training/Adam/sub_47/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_47Subtraining/Adam/sub_47/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_79Multraining/Adam/sub_477training/Adam/gradients/conv4b/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_47Addtraining/Adam/mul_78training/Adam/mul_79*
_output_shapes	
:*
T0
s
training/Adam/mul_80MulAdam/beta_2/readtraining/Adam/Variable_61/read*
T0*
_output_shapes	
:
[
training/Adam/sub_48/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_48Subtraining/Adam/sub_48/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_15Square7training/Adam/gradients/conv4b/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
p
training/Adam/mul_81Multraining/Adam/sub_48training/Adam/Square_15*
T0*
_output_shapes	
:
m
training/Adam/add_48Addtraining/Adam/mul_80training/Adam/mul_81*
T0*
_output_shapes	
:
l
training/Adam/mul_82Multraining/Adam/mul_2training/Adam/add_47*
T0*
_output_shapes	
:
[
training/Adam/Const_32Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_33Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_16/MinimumMinimumtraining/Adam/add_48training/Adam/Const_33*
_output_shapes	
:*
T0

training/Adam/clip_by_value_16Maximum&training/Adam/clip_by_value_16/Minimumtraining/Adam/Const_32*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_16Sqrttraining/Adam/clip_by_value_16*
_output_shapes	
:*
T0
[
training/Adam/add_49/yConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
p
training/Adam/add_49Addtraining/Adam/Sqrt_16training/Adam/add_49/y*
_output_shapes	
:*
T0
u
training/Adam/truediv_17RealDivtraining/Adam/mul_82training/Adam/add_49*
_output_shapes	
:*
T0
m
training/Adam/sub_49Subconv4b/bias/readtraining/Adam/truediv_17*
T0*
_output_shapes	
:
Ď
training/Adam/Assign_45Assigntraining/Adam/Variable_15training/Adam/add_47*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_46Assigntraining/Adam/Variable_61training/Adam/add_48*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_61*
validate_shape(*
_output_shapes	
:
ł
training/Adam/Assign_47Assignconv4b/biastraining/Adam/sub_49*
use_locking(*
T0*
_class
loc:@conv4b/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_83MulAdam/beta_1/readtraining/Adam/Variable_16/read*
T0*(
_output_shapes
:
[
training/Adam/sub_50/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_50Subtraining/Adam/sub_50/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ş
training/Adam/mul_84Multraining/Adam/sub_50Dtraining/Adam/gradients/conv5a/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
z
training/Adam/add_50Addtraining/Adam/mul_83training/Adam/mul_84*(
_output_shapes
:*
T0

training/Adam/mul_85MulAdam/beta_2/readtraining/Adam/Variable_62/read*(
_output_shapes
:*
T0
[
training/Adam/sub_51/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_51Subtraining/Adam/sub_51/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_16SquareDtraining/Adam/gradients/conv5a/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
}
training/Adam/mul_86Multraining/Adam/sub_51training/Adam/Square_16*
T0*(
_output_shapes
:
z
training/Adam/add_51Addtraining/Adam/mul_85training/Adam/mul_86*
T0*(
_output_shapes
:
y
training/Adam/mul_87Multraining/Adam/mul_2training/Adam/add_50*(
_output_shapes
:*
T0
[
training/Adam/Const_34Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_35Const*
dtype0*
_output_shapes
: *
valueB
 *  

&training/Adam/clip_by_value_17/MinimumMinimumtraining/Adam/add_51training/Adam/Const_35*
T0*(
_output_shapes
:

training/Adam/clip_by_value_17Maximum&training/Adam/clip_by_value_17/Minimumtraining/Adam/Const_34*(
_output_shapes
:*
T0
p
training/Adam/Sqrt_17Sqrttraining/Adam/clip_by_value_17*(
_output_shapes
:*
T0
[
training/Adam/add_52/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_52Addtraining/Adam/Sqrt_17training/Adam/add_52/y*
T0*(
_output_shapes
:

training/Adam/truediv_18RealDivtraining/Adam/mul_87training/Adam/add_52*
T0*(
_output_shapes
:
|
training/Adam/sub_52Subconv5a/kernel/readtraining/Adam/truediv_18*
T0*(
_output_shapes
:
Ü
training/Adam/Assign_48Assigntraining/Adam/Variable_16training/Adam/add_50*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_49Assigntraining/Adam/Variable_62training/Adam/add_51*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_62*
validate_shape(*(
_output_shapes
:
Ä
training/Adam/Assign_50Assignconv5a/kerneltraining/Adam/sub_52*
use_locking(*
T0* 
_class
loc:@conv5a/kernel*
validate_shape(*(
_output_shapes
:
s
training/Adam/mul_88MulAdam/beta_1/readtraining/Adam/Variable_17/read*
_output_shapes	
:*
T0
[
training/Adam/sub_53/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_53Subtraining/Adam/sub_53/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_89Multraining/Adam/sub_537training/Adam/gradients/conv5a/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_53Addtraining/Adam/mul_88training/Adam/mul_89*
_output_shapes	
:*
T0
s
training/Adam/mul_90MulAdam/beta_2/readtraining/Adam/Variable_63/read*
T0*
_output_shapes	
:
[
training/Adam/sub_54/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_54Subtraining/Adam/sub_54/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_17Square7training/Adam/gradients/conv5a/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
p
training/Adam/mul_91Multraining/Adam/sub_54training/Adam/Square_17*
_output_shapes	
:*
T0
m
training/Adam/add_54Addtraining/Adam/mul_90training/Adam/mul_91*
T0*
_output_shapes	
:
l
training/Adam/mul_92Multraining/Adam/mul_2training/Adam/add_53*
T0*
_output_shapes	
:
[
training/Adam/Const_36Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_37Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_18/MinimumMinimumtraining/Adam/add_54training/Adam/Const_37*
T0*
_output_shapes	
:

training/Adam/clip_by_value_18Maximum&training/Adam/clip_by_value_18/Minimumtraining/Adam/Const_36*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_18Sqrttraining/Adam/clip_by_value_18*
T0*
_output_shapes	
:
[
training/Adam/add_55/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_55Addtraining/Adam/Sqrt_18training/Adam/add_55/y*
T0*
_output_shapes	
:
u
training/Adam/truediv_19RealDivtraining/Adam/mul_92training/Adam/add_55*
T0*
_output_shapes	
:
m
training/Adam/sub_55Subconv5a/bias/readtraining/Adam/truediv_19*
T0*
_output_shapes	
:
Ď
training/Adam/Assign_51Assigntraining/Adam/Variable_17training/Adam/add_53*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17
Ď
training/Adam/Assign_52Assigntraining/Adam/Variable_63training/Adam/add_54*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_63
ł
training/Adam/Assign_53Assignconv5a/biastraining/Adam/sub_55*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv5a/bias

training/Adam/mul_93MulAdam/beta_1/readtraining/Adam/Variable_18/read*(
_output_shapes
:*
T0
[
training/Adam/sub_56/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_56Subtraining/Adam/sub_56/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ş
training/Adam/mul_94Multraining/Adam/sub_56Dtraining/Adam/gradients/conv5b/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
z
training/Adam/add_56Addtraining/Adam/mul_93training/Adam/mul_94*
T0*(
_output_shapes
:

training/Adam/mul_95MulAdam/beta_2/readtraining/Adam/Variable_64/read*(
_output_shapes
:*
T0
[
training/Adam/sub_57/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_57Subtraining/Adam/sub_57/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_18SquareDtraining/Adam/gradients/conv5b/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
}
training/Adam/mul_96Multraining/Adam/sub_57training/Adam/Square_18*
T0*(
_output_shapes
:
z
training/Adam/add_57Addtraining/Adam/mul_95training/Adam/mul_96*(
_output_shapes
:*
T0
y
training/Adam/mul_97Multraining/Adam/mul_2training/Adam/add_56*(
_output_shapes
:*
T0
[
training/Adam/Const_38Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_39Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_19/MinimumMinimumtraining/Adam/add_57training/Adam/Const_39*
T0*(
_output_shapes
:

training/Adam/clip_by_value_19Maximum&training/Adam/clip_by_value_19/Minimumtraining/Adam/Const_38*(
_output_shapes
:*
T0
p
training/Adam/Sqrt_19Sqrttraining/Adam/clip_by_value_19*(
_output_shapes
:*
T0
[
training/Adam/add_58/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_58Addtraining/Adam/Sqrt_19training/Adam/add_58/y*(
_output_shapes
:*
T0

training/Adam/truediv_20RealDivtraining/Adam/mul_97training/Adam/add_58*(
_output_shapes
:*
T0
|
training/Adam/sub_58Subconv5b/kernel/readtraining/Adam/truediv_20*
T0*(
_output_shapes
:
Ü
training/Adam/Assign_54Assigntraining/Adam/Variable_18training/Adam/add_56*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_55Assigntraining/Adam/Variable_64training/Adam/add_57*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_64
Ä
training/Adam/Assign_56Assignconv5b/kerneltraining/Adam/sub_58*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv5b/kernel
s
training/Adam/mul_98MulAdam/beta_1/readtraining/Adam/Variable_19/read*
_output_shapes	
:*
T0
[
training/Adam/sub_59/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_59Subtraining/Adam/sub_59/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_99Multraining/Adam/sub_597training/Adam/gradients/conv5b/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
m
training/Adam/add_59Addtraining/Adam/mul_98training/Adam/mul_99*
T0*
_output_shapes	
:
t
training/Adam/mul_100MulAdam/beta_2/readtraining/Adam/Variable_65/read*
_output_shapes	
:*
T0
[
training/Adam/sub_60/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_60Subtraining/Adam/sub_60/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_19Square7training/Adam/gradients/conv5b/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
q
training/Adam/mul_101Multraining/Adam/sub_60training/Adam/Square_19*
_output_shapes	
:*
T0
o
training/Adam/add_60Addtraining/Adam/mul_100training/Adam/mul_101*
T0*
_output_shapes	
:
m
training/Adam/mul_102Multraining/Adam/mul_2training/Adam/add_59*
T0*
_output_shapes	
:
[
training/Adam/Const_40Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_41Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_20/MinimumMinimumtraining/Adam/add_60training/Adam/Const_41*
T0*
_output_shapes	
:

training/Adam/clip_by_value_20Maximum&training/Adam/clip_by_value_20/Minimumtraining/Adam/Const_40*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_20Sqrttraining/Adam/clip_by_value_20*
T0*
_output_shapes	
:
[
training/Adam/add_61/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_61Addtraining/Adam/Sqrt_20training/Adam/add_61/y*
_output_shapes	
:*
T0
v
training/Adam/truediv_21RealDivtraining/Adam/mul_102training/Adam/add_61*
_output_shapes	
:*
T0
m
training/Adam/sub_61Subconv5b/bias/readtraining/Adam/truediv_21*
T0*
_output_shapes	
:
Ď
training/Adam/Assign_57Assigntraining/Adam/Variable_19training/Adam/add_59*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes	
:*
use_locking(
Ď
training/Adam/Assign_58Assigntraining/Adam/Variable_65training/Adam/add_60*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_65
ł
training/Adam/Assign_59Assignconv5b/biastraining/Adam/sub_61*
use_locking(*
T0*
_class
loc:@conv5b/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_103MulAdam/beta_1/readtraining/Adam/Variable_20/read*(
_output_shapes
:*
T0
[
training/Adam/sub_62/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_62Subtraining/Adam/sub_62/xAdam/beta_1/read*
_output_shapes
: *
T0
Ş
training/Adam/mul_104Multraining/Adam/sub_62Ctraining/Adam/gradients/conv6/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
|
training/Adam/add_62Addtraining/Adam/mul_103training/Adam/mul_104*
T0*(
_output_shapes
:

training/Adam/mul_105MulAdam/beta_2/readtraining/Adam/Variable_66/read*
T0*(
_output_shapes
:
[
training/Adam/sub_63/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_63Subtraining/Adam/sub_63/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_20SquareCtraining/Adam/gradients/conv6/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
~
training/Adam/mul_106Multraining/Adam/sub_63training/Adam/Square_20*
T0*(
_output_shapes
:
|
training/Adam/add_63Addtraining/Adam/mul_105training/Adam/mul_106*
T0*(
_output_shapes
:
z
training/Adam/mul_107Multraining/Adam/mul_2training/Adam/add_62*
T0*(
_output_shapes
:
[
training/Adam/Const_42Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_43Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_21/MinimumMinimumtraining/Adam/add_63training/Adam/Const_43*(
_output_shapes
:*
T0

training/Adam/clip_by_value_21Maximum&training/Adam/clip_by_value_21/Minimumtraining/Adam/Const_42*
T0*(
_output_shapes
:
p
training/Adam/Sqrt_21Sqrttraining/Adam/clip_by_value_21*
T0*(
_output_shapes
:
[
training/Adam/add_64/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_64Addtraining/Adam/Sqrt_21training/Adam/add_64/y*(
_output_shapes
:*
T0

training/Adam/truediv_22RealDivtraining/Adam/mul_107training/Adam/add_64*
T0*(
_output_shapes
:
{
training/Adam/sub_64Subconv6/kernel/readtraining/Adam/truediv_22*
T0*(
_output_shapes
:
Ü
training/Adam/Assign_60Assigntraining/Adam/Variable_20training/Adam/add_62*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_61Assigntraining/Adam/Variable_66training/Adam/add_63*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_66*
validate_shape(
Â
training/Adam/Assign_62Assignconv6/kerneltraining/Adam/sub_64*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*
_class
loc:@conv6/kernel
t
training/Adam/mul_108MulAdam/beta_1/readtraining/Adam/Variable_21/read*
T0*
_output_shapes	
:
[
training/Adam/sub_65/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_65Subtraining/Adam/sub_65/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_109Multraining/Adam/sub_656training/Adam/gradients/conv6/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/add_65Addtraining/Adam/mul_108training/Adam/mul_109*
_output_shapes	
:*
T0
t
training/Adam/mul_110MulAdam/beta_2/readtraining/Adam/Variable_67/read*
T0*
_output_shapes	
:
[
training/Adam/sub_66/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_66Subtraining/Adam/sub_66/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_21Square6training/Adam/gradients/conv6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
q
training/Adam/mul_111Multraining/Adam/sub_66training/Adam/Square_21*
_output_shapes	
:*
T0
o
training/Adam/add_66Addtraining/Adam/mul_110training/Adam/mul_111*
_output_shapes	
:*
T0
m
training/Adam/mul_112Multraining/Adam/mul_2training/Adam/add_65*
_output_shapes	
:*
T0
[
training/Adam/Const_44Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_45Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_22/MinimumMinimumtraining/Adam/add_66training/Adam/Const_45*
T0*
_output_shapes	
:

training/Adam/clip_by_value_22Maximum&training/Adam/clip_by_value_22/Minimumtraining/Adam/Const_44*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_22Sqrttraining/Adam/clip_by_value_22*
T0*
_output_shapes	
:
[
training/Adam/add_67/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_67Addtraining/Adam/Sqrt_22training/Adam/add_67/y*
T0*
_output_shapes	
:
v
training/Adam/truediv_23RealDivtraining/Adam/mul_112training/Adam/add_67*
T0*
_output_shapes	
:
l
training/Adam/sub_67Subconv6/bias/readtraining/Adam/truediv_23*
T0*
_output_shapes	
:
Ď
training/Adam/Assign_63Assigntraining/Adam/Variable_21training/Adam/add_65*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_64Assigntraining/Adam/Variable_67training/Adam/add_66*,
_class"
 loc:@training/Adam/Variable_67*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ą
training/Adam/Assign_65Assign
conv6/biastraining/Adam/sub_67*
use_locking(*
T0*
_class
loc:@conv6/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_113MulAdam/beta_1/readtraining/Adam/Variable_22/read*
T0*(
_output_shapes
:
[
training/Adam/sub_68/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_68Subtraining/Adam/sub_68/xAdam/beta_1/read*
_output_shapes
: *
T0
Ť
training/Adam/mul_114Multraining/Adam/sub_68Dtraining/Adam/gradients/conv7a/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
|
training/Adam/add_68Addtraining/Adam/mul_113training/Adam/mul_114*
T0*(
_output_shapes
:

training/Adam/mul_115MulAdam/beta_2/readtraining/Adam/Variable_68/read*(
_output_shapes
:*
T0
[
training/Adam/sub_69/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_69Subtraining/Adam/sub_69/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_22SquareDtraining/Adam/gradients/conv7a/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
~
training/Adam/mul_116Multraining/Adam/sub_69training/Adam/Square_22*
T0*(
_output_shapes
:
|
training/Adam/add_69Addtraining/Adam/mul_115training/Adam/mul_116*(
_output_shapes
:*
T0
z
training/Adam/mul_117Multraining/Adam/mul_2training/Adam/add_68*
T0*(
_output_shapes
:
[
training/Adam/Const_46Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_47Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_23/MinimumMinimumtraining/Adam/add_69training/Adam/Const_47*
T0*(
_output_shapes
:

training/Adam/clip_by_value_23Maximum&training/Adam/clip_by_value_23/Minimumtraining/Adam/Const_46*(
_output_shapes
:*
T0
p
training/Adam/Sqrt_23Sqrttraining/Adam/clip_by_value_23*
T0*(
_output_shapes
:
[
training/Adam/add_70/yConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
}
training/Adam/add_70Addtraining/Adam/Sqrt_23training/Adam/add_70/y*
T0*(
_output_shapes
:

training/Adam/truediv_24RealDivtraining/Adam/mul_117training/Adam/add_70*
T0*(
_output_shapes
:
|
training/Adam/sub_70Subconv7a/kernel/readtraining/Adam/truediv_24*
T0*(
_output_shapes
:
Ü
training/Adam/Assign_66Assigntraining/Adam/Variable_22training/Adam/add_68*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_67Assigntraining/Adam/Variable_68training/Adam/add_69*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_68
Ä
training/Adam/Assign_68Assignconv7a/kerneltraining/Adam/sub_70*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv7a/kernel
t
training/Adam/mul_118MulAdam/beta_1/readtraining/Adam/Variable_23/read*
T0*
_output_shapes	
:
[
training/Adam/sub_71/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_71Subtraining/Adam/sub_71/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_119Multraining/Adam/sub_717training/Adam/gradients/conv7a/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/add_71Addtraining/Adam/mul_118training/Adam/mul_119*
_output_shapes	
:*
T0
t
training/Adam/mul_120MulAdam/beta_2/readtraining/Adam/Variable_69/read*
_output_shapes	
:*
T0
[
training/Adam/sub_72/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_72Subtraining/Adam/sub_72/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_23Square7training/Adam/gradients/conv7a/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
q
training/Adam/mul_121Multraining/Adam/sub_72training/Adam/Square_23*
_output_shapes	
:*
T0
o
training/Adam/add_72Addtraining/Adam/mul_120training/Adam/mul_121*
_output_shapes	
:*
T0
m
training/Adam/mul_122Multraining/Adam/mul_2training/Adam/add_71*
_output_shapes	
:*
T0
[
training/Adam/Const_48Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_49Const*
dtype0*
_output_shapes
: *
valueB
 *  

&training/Adam/clip_by_value_24/MinimumMinimumtraining/Adam/add_72training/Adam/Const_49*
T0*
_output_shapes	
:

training/Adam/clip_by_value_24Maximum&training/Adam/clip_by_value_24/Minimumtraining/Adam/Const_48*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_24Sqrttraining/Adam/clip_by_value_24*
_output_shapes	
:*
T0
[
training/Adam/add_73/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_73Addtraining/Adam/Sqrt_24training/Adam/add_73/y*
T0*
_output_shapes	
:
v
training/Adam/truediv_25RealDivtraining/Adam/mul_122training/Adam/add_73*
_output_shapes	
:*
T0
m
training/Adam/sub_73Subconv7a/bias/readtraining/Adam/truediv_25*
_output_shapes	
:*
T0
Ď
training/Adam/Assign_69Assigntraining/Adam/Variable_23training/Adam/add_71*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_70Assigntraining/Adam/Variable_69training/Adam/add_72*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_69*
validate_shape(*
_output_shapes	
:
ł
training/Adam/Assign_71Assignconv7a/biastraining/Adam/sub_73*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv7a/bias

training/Adam/mul_123MulAdam/beta_1/readtraining/Adam/Variable_24/read*
T0*(
_output_shapes
:
[
training/Adam/sub_74/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_74Subtraining/Adam/sub_74/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ť
training/Adam/mul_124Multraining/Adam/sub_74Dtraining/Adam/gradients/conv7b/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
|
training/Adam/add_74Addtraining/Adam/mul_123training/Adam/mul_124*(
_output_shapes
:*
T0

training/Adam/mul_125MulAdam/beta_2/readtraining/Adam/Variable_70/read*
T0*(
_output_shapes
:
[
training/Adam/sub_75/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_75Subtraining/Adam/sub_75/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_24SquareDtraining/Adam/gradients/conv7b/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
~
training/Adam/mul_126Multraining/Adam/sub_75training/Adam/Square_24*
T0*(
_output_shapes
:
|
training/Adam/add_75Addtraining/Adam/mul_125training/Adam/mul_126*
T0*(
_output_shapes
:
z
training/Adam/mul_127Multraining/Adam/mul_2training/Adam/add_74*(
_output_shapes
:*
T0
[
training/Adam/Const_50Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_51Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_25/MinimumMinimumtraining/Adam/add_75training/Adam/Const_51*
T0*(
_output_shapes
:

training/Adam/clip_by_value_25Maximum&training/Adam/clip_by_value_25/Minimumtraining/Adam/Const_50*(
_output_shapes
:*
T0
p
training/Adam/Sqrt_25Sqrttraining/Adam/clip_by_value_25*
T0*(
_output_shapes
:
[
training/Adam/add_76/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_76Addtraining/Adam/Sqrt_25training/Adam/add_76/y*(
_output_shapes
:*
T0

training/Adam/truediv_26RealDivtraining/Adam/mul_127training/Adam/add_76*
T0*(
_output_shapes
:
|
training/Adam/sub_76Subconv7b/kernel/readtraining/Adam/truediv_26*(
_output_shapes
:*
T0
Ü
training/Adam/Assign_72Assigntraining/Adam/Variable_24training/Adam/add_74*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_73Assigntraining/Adam/Variable_70training/Adam/add_75*,
_class"
 loc:@training/Adam/Variable_70*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Ä
training/Adam/Assign_74Assignconv7b/kerneltraining/Adam/sub_76* 
_class
loc:@conv7b/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
t
training/Adam/mul_128MulAdam/beta_1/readtraining/Adam/Variable_25/read*
T0*
_output_shapes	
:
[
training/Adam/sub_77/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_77Subtraining/Adam/sub_77/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_129Multraining/Adam/sub_777training/Adam/gradients/conv7b/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/add_77Addtraining/Adam/mul_128training/Adam/mul_129*
T0*
_output_shapes	
:
t
training/Adam/mul_130MulAdam/beta_2/readtraining/Adam/Variable_71/read*
T0*
_output_shapes	
:
[
training/Adam/sub_78/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_78Subtraining/Adam/sub_78/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_25Square7training/Adam/gradients/conv7b/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
q
training/Adam/mul_131Multraining/Adam/sub_78training/Adam/Square_25*
T0*
_output_shapes	
:
o
training/Adam/add_78Addtraining/Adam/mul_130training/Adam/mul_131*
T0*
_output_shapes	
:
m
training/Adam/mul_132Multraining/Adam/mul_2training/Adam/add_77*
T0*
_output_shapes	
:
[
training/Adam/Const_52Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_53Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_26/MinimumMinimumtraining/Adam/add_78training/Adam/Const_53*
T0*
_output_shapes	
:

training/Adam/clip_by_value_26Maximum&training/Adam/clip_by_value_26/Minimumtraining/Adam/Const_52*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_26Sqrttraining/Adam/clip_by_value_26*
_output_shapes	
:*
T0
[
training/Adam/add_79/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_79Addtraining/Adam/Sqrt_26training/Adam/add_79/y*
T0*
_output_shapes	
:
v
training/Adam/truediv_27RealDivtraining/Adam/mul_132training/Adam/add_79*
T0*
_output_shapes	
:
m
training/Adam/sub_79Subconv7b/bias/readtraining/Adam/truediv_27*
T0*
_output_shapes	
:
Ď
training/Adam/Assign_75Assigntraining/Adam/Variable_25training/Adam/add_77*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_76Assigntraining/Adam/Variable_71training/Adam/add_78*,
_class"
 loc:@training/Adam/Variable_71*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ł
training/Adam/Assign_77Assignconv7b/biastraining/Adam/sub_79*
use_locking(*
T0*
_class
loc:@conv7b/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_133MulAdam/beta_1/readtraining/Adam/Variable_26/read*
T0*(
_output_shapes
:
[
training/Adam/sub_80/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_80Subtraining/Adam/sub_80/xAdam/beta_1/read*
T0*
_output_shapes
: 
­
training/Adam/mul_134Multraining/Adam/sub_80Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
|
training/Adam/add_80Addtraining/Adam/mul_133training/Adam/mul_134*(
_output_shapes
:*
T0

training/Adam/mul_135MulAdam/beta_2/readtraining/Adam/Variable_72/read*
T0*(
_output_shapes
:
[
training/Adam/sub_81/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_81Subtraining/Adam/sub_81/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_26SquareFtraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
~
training/Adam/mul_136Multraining/Adam/sub_81training/Adam/Square_26*
T0*(
_output_shapes
:
|
training/Adam/add_81Addtraining/Adam/mul_135training/Adam/mul_136*
T0*(
_output_shapes
:
z
training/Adam/mul_137Multraining/Adam/mul_2training/Adam/add_80*
T0*(
_output_shapes
:
[
training/Adam/Const_54Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_55Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_27/MinimumMinimumtraining/Adam/add_81training/Adam/Const_55*
T0*(
_output_shapes
:

training/Adam/clip_by_value_27Maximum&training/Adam/clip_by_value_27/Minimumtraining/Adam/Const_54*(
_output_shapes
:*
T0
p
training/Adam/Sqrt_27Sqrttraining/Adam/clip_by_value_27*
T0*(
_output_shapes
:
[
training/Adam/add_82/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_82Addtraining/Adam/Sqrt_27training/Adam/add_82/y*(
_output_shapes
:*
T0

training/Adam/truediv_28RealDivtraining/Adam/mul_137training/Adam/add_82*
T0*(
_output_shapes
:
~
training/Adam/sub_82Subconv2d_1/kernel/readtraining/Adam/truediv_28*(
_output_shapes
:*
T0
Ü
training/Adam/Assign_78Assigntraining/Adam/Variable_26training/Adam/add_80*
T0*,
_class"
 loc:@training/Adam/Variable_26*
validate_shape(*(
_output_shapes
:*
use_locking(
Ü
training/Adam/Assign_79Assigntraining/Adam/Variable_72training/Adam/add_81*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_72*
validate_shape(*(
_output_shapes
:
Č
training/Adam/Assign_80Assignconv2d_1/kerneltraining/Adam/sub_82*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
t
training/Adam/mul_138MulAdam/beta_1/readtraining/Adam/Variable_27/read*
T0*
_output_shapes	
:
[
training/Adam/sub_83/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_83Subtraining/Adam/sub_83/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_139Multraining/Adam/sub_839training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/add_83Addtraining/Adam/mul_138training/Adam/mul_139*
T0*
_output_shapes	
:
t
training/Adam/mul_140MulAdam/beta_2/readtraining/Adam/Variable_73/read*
T0*
_output_shapes	
:
[
training/Adam/sub_84/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_84Subtraining/Adam/sub_84/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_27Square9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
q
training/Adam/mul_141Multraining/Adam/sub_84training/Adam/Square_27*
T0*
_output_shapes	
:
o
training/Adam/add_84Addtraining/Adam/mul_140training/Adam/mul_141*
_output_shapes	
:*
T0
m
training/Adam/mul_142Multraining/Adam/mul_2training/Adam/add_83*
_output_shapes	
:*
T0
[
training/Adam/Const_56Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_57Const*
dtype0*
_output_shapes
: *
valueB
 *  

&training/Adam/clip_by_value_28/MinimumMinimumtraining/Adam/add_84training/Adam/Const_57*
T0*
_output_shapes	
:

training/Adam/clip_by_value_28Maximum&training/Adam/clip_by_value_28/Minimumtraining/Adam/Const_56*
_output_shapes	
:*
T0
c
training/Adam/Sqrt_28Sqrttraining/Adam/clip_by_value_28*
_output_shapes	
:*
T0
[
training/Adam/add_85/yConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
p
training/Adam/add_85Addtraining/Adam/Sqrt_28training/Adam/add_85/y*
T0*
_output_shapes	
:
v
training/Adam/truediv_29RealDivtraining/Adam/mul_142training/Adam/add_85*
_output_shapes	
:*
T0
o
training/Adam/sub_85Subconv2d_1/bias/readtraining/Adam/truediv_29*
T0*
_output_shapes	
:
Ď
training/Adam/Assign_81Assigntraining/Adam/Variable_27training/Adam/add_83*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_27*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_82Assigntraining/Adam/Variable_73training/Adam/add_84*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_73*
validate_shape(
ˇ
training/Adam/Assign_83Assignconv2d_1/biastraining/Adam/sub_85*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes	
:

training/Adam/mul_143MulAdam/beta_1/readtraining/Adam/Variable_28/read*(
_output_shapes
:*
T0
[
training/Adam/sub_86/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_86Subtraining/Adam/sub_86/xAdam/beta_1/read*
_output_shapes
: *
T0
­
training/Adam/mul_144Multraining/Adam/sub_86Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
|
training/Adam/add_86Addtraining/Adam/mul_143training/Adam/mul_144*(
_output_shapes
:*
T0

training/Adam/mul_145MulAdam/beta_2/readtraining/Adam/Variable_74/read*(
_output_shapes
:*
T0
[
training/Adam/sub_87/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_87Subtraining/Adam/sub_87/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_28SquareFtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
~
training/Adam/mul_146Multraining/Adam/sub_87training/Adam/Square_28*
T0*(
_output_shapes
:
|
training/Adam/add_87Addtraining/Adam/mul_145training/Adam/mul_146*
T0*(
_output_shapes
:
z
training/Adam/mul_147Multraining/Adam/mul_2training/Adam/add_86*
T0*(
_output_shapes
:
[
training/Adam/Const_58Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_59Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_29/MinimumMinimumtraining/Adam/add_87training/Adam/Const_59*(
_output_shapes
:*
T0

training/Adam/clip_by_value_29Maximum&training/Adam/clip_by_value_29/Minimumtraining/Adam/Const_58*(
_output_shapes
:*
T0
p
training/Adam/Sqrt_29Sqrttraining/Adam/clip_by_value_29*
T0*(
_output_shapes
:
[
training/Adam/add_88/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_88Addtraining/Adam/Sqrt_29training/Adam/add_88/y*
T0*(
_output_shapes
:

training/Adam/truediv_30RealDivtraining/Adam/mul_147training/Adam/add_88*
T0*(
_output_shapes
:
~
training/Adam/sub_88Subconv2d_2/kernel/readtraining/Adam/truediv_30*(
_output_shapes
:*
T0
Ü
training/Adam/Assign_84Assigntraining/Adam/Variable_28training/Adam/add_86*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_28*
validate_shape(
Ü
training/Adam/Assign_85Assigntraining/Adam/Variable_74training/Adam/add_87*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_74*
validate_shape(*(
_output_shapes
:
Č
training/Adam/Assign_86Assignconv2d_2/kerneltraining/Adam/sub_88*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*(
_output_shapes
:
t
training/Adam/mul_148MulAdam/beta_1/readtraining/Adam/Variable_29/read*
_output_shapes	
:*
T0
[
training/Adam/sub_89/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_89Subtraining/Adam/sub_89/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_149Multraining/Adam/sub_899training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/add_89Addtraining/Adam/mul_148training/Adam/mul_149*
_output_shapes	
:*
T0
t
training/Adam/mul_150MulAdam/beta_2/readtraining/Adam/Variable_75/read*
T0*
_output_shapes	
:
[
training/Adam/sub_90/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_90Subtraining/Adam/sub_90/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_29Square9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
q
training/Adam/mul_151Multraining/Adam/sub_90training/Adam/Square_29*
T0*
_output_shapes	
:
o
training/Adam/add_90Addtraining/Adam/mul_150training/Adam/mul_151*
_output_shapes	
:*
T0
m
training/Adam/mul_152Multraining/Adam/mul_2training/Adam/add_89*
_output_shapes	
:*
T0
[
training/Adam/Const_60Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_61Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_30/MinimumMinimumtraining/Adam/add_90training/Adam/Const_61*
T0*
_output_shapes	
:

training/Adam/clip_by_value_30Maximum&training/Adam/clip_by_value_30/Minimumtraining/Adam/Const_60*
_output_shapes	
:*
T0
c
training/Adam/Sqrt_30Sqrttraining/Adam/clip_by_value_30*
T0*
_output_shapes	
:
[
training/Adam/add_91/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_91Addtraining/Adam/Sqrt_30training/Adam/add_91/y*
_output_shapes	
:*
T0
v
training/Adam/truediv_31RealDivtraining/Adam/mul_152training/Adam/add_91*
_output_shapes	
:*
T0
o
training/Adam/sub_91Subconv2d_2/bias/readtraining/Adam/truediv_31*
_output_shapes	
:*
T0
Ď
training/Adam/Assign_87Assigntraining/Adam/Variable_29training/Adam/add_89*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_29*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_88Assigntraining/Adam/Variable_75training/Adam/add_90*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_75*
validate_shape(*
_output_shapes	
:
ˇ
training/Adam/Assign_89Assignconv2d_2/biastraining/Adam/sub_91* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

training/Adam/mul_153MulAdam/beta_1/readtraining/Adam/Variable_30/read*(
_output_shapes
:*
T0
[
training/Adam/sub_92/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_92Subtraining/Adam/sub_92/xAdam/beta_1/read*
T0*
_output_shapes
: 
­
training/Adam/mul_154Multraining/Adam/sub_92Ftraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
|
training/Adam/add_92Addtraining/Adam/mul_153training/Adam/mul_154*
T0*(
_output_shapes
:

training/Adam/mul_155MulAdam/beta_2/readtraining/Adam/Variable_76/read*
T0*(
_output_shapes
:
[
training/Adam/sub_93/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_93Subtraining/Adam/sub_93/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_30SquareFtraining/Adam/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
~
training/Adam/mul_156Multraining/Adam/sub_93training/Adam/Square_30*
T0*(
_output_shapes
:
|
training/Adam/add_93Addtraining/Adam/mul_155training/Adam/mul_156*(
_output_shapes
:*
T0
z
training/Adam/mul_157Multraining/Adam/mul_2training/Adam/add_92*
T0*(
_output_shapes
:
[
training/Adam/Const_62Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_63Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_31/MinimumMinimumtraining/Adam/add_93training/Adam/Const_63*
T0*(
_output_shapes
:

training/Adam/clip_by_value_31Maximum&training/Adam/clip_by_value_31/Minimumtraining/Adam/Const_62*
T0*(
_output_shapes
:
p
training/Adam/Sqrt_31Sqrttraining/Adam/clip_by_value_31*
T0*(
_output_shapes
:
[
training/Adam/add_94/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_94Addtraining/Adam/Sqrt_31training/Adam/add_94/y*
T0*(
_output_shapes
:

training/Adam/truediv_32RealDivtraining/Adam/mul_157training/Adam/add_94*
T0*(
_output_shapes
:
~
training/Adam/sub_94Subconv2d_3/kernel/readtraining/Adam/truediv_32*
T0*(
_output_shapes
:
Ü
training/Adam/Assign_90Assigntraining/Adam/Variable_30training/Adam/add_92*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_30*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_91Assigntraining/Adam/Variable_76training/Adam/add_93*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_76*
validate_shape(*(
_output_shapes
:
Č
training/Adam/Assign_92Assignconv2d_3/kerneltraining/Adam/sub_94*"
_class
loc:@conv2d_3/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
t
training/Adam/mul_158MulAdam/beta_1/readtraining/Adam/Variable_31/read*
_output_shapes	
:*
T0
[
training/Adam/sub_95/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_95Subtraining/Adam/sub_95/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_159Multraining/Adam/sub_959training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/add_95Addtraining/Adam/mul_158training/Adam/mul_159*
T0*
_output_shapes	
:
t
training/Adam/mul_160MulAdam/beta_2/readtraining/Adam/Variable_77/read*
_output_shapes	
:*
T0
[
training/Adam/sub_96/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_96Subtraining/Adam/sub_96/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_31Square9training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
q
training/Adam/mul_161Multraining/Adam/sub_96training/Adam/Square_31*
_output_shapes	
:*
T0
o
training/Adam/add_96Addtraining/Adam/mul_160training/Adam/mul_161*
_output_shapes	
:*
T0
m
training/Adam/mul_162Multraining/Adam/mul_2training/Adam/add_95*
T0*
_output_shapes	
:
[
training/Adam/Const_64Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_65Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_32/MinimumMinimumtraining/Adam/add_96training/Adam/Const_65*
T0*
_output_shapes	
:

training/Adam/clip_by_value_32Maximum&training/Adam/clip_by_value_32/Minimumtraining/Adam/Const_64*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_32Sqrttraining/Adam/clip_by_value_32*
T0*
_output_shapes	
:
[
training/Adam/add_97/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
p
training/Adam/add_97Addtraining/Adam/Sqrt_32training/Adam/add_97/y*
T0*
_output_shapes	
:
v
training/Adam/truediv_33RealDivtraining/Adam/mul_162training/Adam/add_97*
_output_shapes	
:*
T0
o
training/Adam/sub_97Subconv2d_3/bias/readtraining/Adam/truediv_33*
_output_shapes	
:*
T0
Ď
training/Adam/Assign_93Assigntraining/Adam/Variable_31training/Adam/add_95*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_31*
validate_shape(*
_output_shapes	
:
Ď
training/Adam/Assign_94Assigntraining/Adam/Variable_77training/Adam/add_96*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_77*
validate_shape(*
_output_shapes	
:
ˇ
training/Adam/Assign_95Assignconv2d_3/biastraining/Adam/sub_97* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

training/Adam/mul_163MulAdam/beta_1/readtraining/Adam/Variable_32/read*
T0*(
_output_shapes
:
[
training/Adam/sub_98/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_98Subtraining/Adam/sub_98/xAdam/beta_1/read*
T0*
_output_shapes
: 
­
training/Adam/mul_164Multraining/Adam/sub_98Ftraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
|
training/Adam/add_98Addtraining/Adam/mul_163training/Adam/mul_164*
T0*(
_output_shapes
:

training/Adam/mul_165MulAdam/beta_2/readtraining/Adam/Variable_78/read*
T0*(
_output_shapes
:
[
training/Adam/sub_99/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_99Subtraining/Adam/sub_99/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_32SquareFtraining/Adam/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
~
training/Adam/mul_166Multraining/Adam/sub_99training/Adam/Square_32*(
_output_shapes
:*
T0
|
training/Adam/add_99Addtraining/Adam/mul_165training/Adam/mul_166*
T0*(
_output_shapes
:
z
training/Adam/mul_167Multraining/Adam/mul_2training/Adam/add_98*
T0*(
_output_shapes
:
[
training/Adam/Const_66Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_67Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_33/MinimumMinimumtraining/Adam/add_99training/Adam/Const_67*
T0*(
_output_shapes
:

training/Adam/clip_by_value_33Maximum&training/Adam/clip_by_value_33/Minimumtraining/Adam/Const_66*(
_output_shapes
:*
T0
p
training/Adam/Sqrt_33Sqrttraining/Adam/clip_by_value_33*(
_output_shapes
:*
T0
\
training/Adam/add_100/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 

training/Adam/add_100Addtraining/Adam/Sqrt_33training/Adam/add_100/y*(
_output_shapes
:*
T0

training/Adam/truediv_34RealDivtraining/Adam/mul_167training/Adam/add_100*
T0*(
_output_shapes
:

training/Adam/sub_100Subconv2d_4/kernel/readtraining/Adam/truediv_34*(
_output_shapes
:*
T0
Ü
training/Adam/Assign_96Assigntraining/Adam/Variable_32training/Adam/add_98*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_32*
validate_shape(*(
_output_shapes
:
Ü
training/Adam/Assign_97Assigntraining/Adam/Variable_78training/Adam/add_99*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_78
É
training/Adam/Assign_98Assignconv2d_4/kerneltraining/Adam/sub_100*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*(
_output_shapes
:
t
training/Adam/mul_168MulAdam/beta_1/readtraining/Adam/Variable_33/read*
_output_shapes	
:*
T0
\
training/Adam/sub_101/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_101Subtraining/Adam/sub_101/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_169Multraining/Adam/sub_1019training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
p
training/Adam/add_101Addtraining/Adam/mul_168training/Adam/mul_169*
T0*
_output_shapes	
:
t
training/Adam/mul_170MulAdam/beta_2/readtraining/Adam/Variable_79/read*
T0*
_output_shapes	
:
\
training/Adam/sub_102/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
h
training/Adam/sub_102Subtraining/Adam/sub_102/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_33Square9training/Adam/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
r
training/Adam/mul_171Multraining/Adam/sub_102training/Adam/Square_33*
T0*
_output_shapes	
:
p
training/Adam/add_102Addtraining/Adam/mul_170training/Adam/mul_171*
T0*
_output_shapes	
:
n
training/Adam/mul_172Multraining/Adam/mul_2training/Adam/add_101*
_output_shapes	
:*
T0
[
training/Adam/Const_68Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_69Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_34/MinimumMinimumtraining/Adam/add_102training/Adam/Const_69*
T0*
_output_shapes	
:

training/Adam/clip_by_value_34Maximum&training/Adam/clip_by_value_34/Minimumtraining/Adam/Const_68*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_34Sqrttraining/Adam/clip_by_value_34*
T0*
_output_shapes	
:
\
training/Adam/add_103/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
r
training/Adam/add_103Addtraining/Adam/Sqrt_34training/Adam/add_103/y*
T0*
_output_shapes	
:
w
training/Adam/truediv_35RealDivtraining/Adam/mul_172training/Adam/add_103*
T0*
_output_shapes	
:
p
training/Adam/sub_103Subconv2d_4/bias/readtraining/Adam/truediv_35*
_output_shapes	
:*
T0
Đ
training/Adam/Assign_99Assigntraining/Adam/Variable_33training/Adam/add_101*
T0*,
_class"
 loc:@training/Adam/Variable_33*
validate_shape(*
_output_shapes	
:*
use_locking(
Ń
training/Adam/Assign_100Assigntraining/Adam/Variable_79training/Adam/add_102*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_79*
validate_shape(*
_output_shapes	
:
š
training/Adam/Assign_101Assignconv2d_4/biastraining/Adam/sub_103*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@conv2d_4/bias

training/Adam/mul_173MulAdam/beta_1/readtraining/Adam/Variable_34/read*
T0*(
_output_shapes
:
\
training/Adam/sub_104/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
h
training/Adam/sub_104Subtraining/Adam/sub_104/xAdam/beta_1/read*
_output_shapes
: *
T0
Ž
training/Adam/mul_174Multraining/Adam/sub_104Ftraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:
}
training/Adam/add_104Addtraining/Adam/mul_173training/Adam/mul_174*
T0*(
_output_shapes
:

training/Adam/mul_175MulAdam/beta_2/readtraining/Adam/Variable_80/read*(
_output_shapes
:*
T0
\
training/Adam/sub_105/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_105Subtraining/Adam/sub_105/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_34SquareFtraining/Adam/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0

training/Adam/mul_176Multraining/Adam/sub_105training/Adam/Square_34*(
_output_shapes
:*
T0
}
training/Adam/add_105Addtraining/Adam/mul_175training/Adam/mul_176*
T0*(
_output_shapes
:
{
training/Adam/mul_177Multraining/Adam/mul_2training/Adam/add_104*
T0*(
_output_shapes
:
[
training/Adam/Const_70Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_71Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_35/MinimumMinimumtraining/Adam/add_105training/Adam/Const_71*(
_output_shapes
:*
T0

training/Adam/clip_by_value_35Maximum&training/Adam/clip_by_value_35/Minimumtraining/Adam/Const_70*(
_output_shapes
:*
T0
p
training/Adam/Sqrt_35Sqrttraining/Adam/clip_by_value_35*
T0*(
_output_shapes
:
\
training/Adam/add_106/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 

training/Adam/add_106Addtraining/Adam/Sqrt_35training/Adam/add_106/y*(
_output_shapes
:*
T0

training/Adam/truediv_36RealDivtraining/Adam/mul_177training/Adam/add_106*
T0*(
_output_shapes
:

training/Adam/sub_106Subconv2d_5/kernel/readtraining/Adam/truediv_36*
T0*(
_output_shapes
:
Ţ
training/Adam/Assign_102Assigntraining/Adam/Variable_34training/Adam/add_104*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_34*
validate_shape(*(
_output_shapes
:
Ţ
training/Adam/Assign_103Assigntraining/Adam/Variable_80training/Adam/add_105*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_80*
validate_shape(*(
_output_shapes
:
Ę
training/Adam/Assign_104Assignconv2d_5/kerneltraining/Adam/sub_106*"
_class
loc:@conv2d_5/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
t
training/Adam/mul_178MulAdam/beta_1/readtraining/Adam/Variable_35/read*
T0*
_output_shapes	
:
\
training/Adam/sub_107/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
h
training/Adam/sub_107Subtraining/Adam/sub_107/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_179Multraining/Adam/sub_1079training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
p
training/Adam/add_107Addtraining/Adam/mul_178training/Adam/mul_179*
T0*
_output_shapes	
:
t
training/Adam/mul_180MulAdam/beta_2/readtraining/Adam/Variable_81/read*
T0*
_output_shapes	
:
\
training/Adam/sub_108/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
h
training/Adam/sub_108Subtraining/Adam/sub_108/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_35Square9training/Adam/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
r
training/Adam/mul_181Multraining/Adam/sub_108training/Adam/Square_35*
_output_shapes	
:*
T0
p
training/Adam/add_108Addtraining/Adam/mul_180training/Adam/mul_181*
_output_shapes	
:*
T0
n
training/Adam/mul_182Multraining/Adam/mul_2training/Adam/add_107*
T0*
_output_shapes	
:
[
training/Adam/Const_72Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_73Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_36/MinimumMinimumtraining/Adam/add_108training/Adam/Const_73*
_output_shapes	
:*
T0

training/Adam/clip_by_value_36Maximum&training/Adam/clip_by_value_36/Minimumtraining/Adam/Const_72*
_output_shapes	
:*
T0
c
training/Adam/Sqrt_36Sqrttraining/Adam/clip_by_value_36*
_output_shapes	
:*
T0
\
training/Adam/add_109/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
r
training/Adam/add_109Addtraining/Adam/Sqrt_36training/Adam/add_109/y*
_output_shapes	
:*
T0
w
training/Adam/truediv_37RealDivtraining/Adam/mul_182training/Adam/add_109*
T0*
_output_shapes	
:
p
training/Adam/sub_109Subconv2d_5/bias/readtraining/Adam/truediv_37*
_output_shapes	
:*
T0
Ń
training/Adam/Assign_105Assigntraining/Adam/Variable_35training/Adam/add_107*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_35
Ń
training/Adam/Assign_106Assigntraining/Adam/Variable_81training/Adam/add_108*
T0*,
_class"
 loc:@training/Adam/Variable_81*
validate_shape(*
_output_shapes	
:*
use_locking(
š
training/Adam/Assign_107Assignconv2d_5/biastraining/Adam/sub_109*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias*
validate_shape(

training/Adam/mul_183MulAdam/beta_1/readtraining/Adam/Variable_36/read*
T0*(
_output_shapes
:
\
training/Adam/sub_110/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_110Subtraining/Adam/sub_110/xAdam/beta_1/read*
_output_shapes
: *
T0
Ž
training/Adam/mul_184Multraining/Adam/sub_110Ftraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
}
training/Adam/add_110Addtraining/Adam/mul_183training/Adam/mul_184*(
_output_shapes
:*
T0

training/Adam/mul_185MulAdam/beta_2/readtraining/Adam/Variable_82/read*
T0*(
_output_shapes
:
\
training/Adam/sub_111/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_111Subtraining/Adam/sub_111/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_36SquareFtraining/Adam/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0

training/Adam/mul_186Multraining/Adam/sub_111training/Adam/Square_36*(
_output_shapes
:*
T0
}
training/Adam/add_111Addtraining/Adam/mul_185training/Adam/mul_186*
T0*(
_output_shapes
:
{
training/Adam/mul_187Multraining/Adam/mul_2training/Adam/add_110*(
_output_shapes
:*
T0
[
training/Adam/Const_74Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_75Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_37/MinimumMinimumtraining/Adam/add_111training/Adam/Const_75*
T0*(
_output_shapes
:

training/Adam/clip_by_value_37Maximum&training/Adam/clip_by_value_37/Minimumtraining/Adam/Const_74*(
_output_shapes
:*
T0
p
training/Adam/Sqrt_37Sqrttraining/Adam/clip_by_value_37*
T0*(
_output_shapes
:
\
training/Adam/add_112/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 

training/Adam/add_112Addtraining/Adam/Sqrt_37training/Adam/add_112/y*
T0*(
_output_shapes
:

training/Adam/truediv_38RealDivtraining/Adam/mul_187training/Adam/add_112*(
_output_shapes
:*
T0

training/Adam/sub_112Subconv2d_6/kernel/readtraining/Adam/truediv_38*
T0*(
_output_shapes
:
Ţ
training/Adam/Assign_108Assigntraining/Adam/Variable_36training/Adam/add_110*
T0*,
_class"
 loc:@training/Adam/Variable_36*
validate_shape(*(
_output_shapes
:*
use_locking(
Ţ
training/Adam/Assign_109Assigntraining/Adam/Variable_82training/Adam/add_111*,
_class"
 loc:@training/Adam/Variable_82*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Ę
training/Adam/Assign_110Assignconv2d_6/kerneltraining/Adam/sub_112*(
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(
t
training/Adam/mul_188MulAdam/beta_1/readtraining/Adam/Variable_37/read*
T0*
_output_shapes	
:
\
training/Adam/sub_113/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_113Subtraining/Adam/sub_113/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_189Multraining/Adam/sub_1139training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
p
training/Adam/add_113Addtraining/Adam/mul_188training/Adam/mul_189*
_output_shapes	
:*
T0
t
training/Adam/mul_190MulAdam/beta_2/readtraining/Adam/Variable_83/read*
T0*
_output_shapes	
:
\
training/Adam/sub_114/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_114Subtraining/Adam/sub_114/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_37Square9training/Adam/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
r
training/Adam/mul_191Multraining/Adam/sub_114training/Adam/Square_37*
T0*
_output_shapes	
:
p
training/Adam/add_114Addtraining/Adam/mul_190training/Adam/mul_191*
T0*
_output_shapes	
:
n
training/Adam/mul_192Multraining/Adam/mul_2training/Adam/add_113*
_output_shapes	
:*
T0
[
training/Adam/Const_76Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_77Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_38/MinimumMinimumtraining/Adam/add_114training/Adam/Const_77*
T0*
_output_shapes	
:

training/Adam/clip_by_value_38Maximum&training/Adam/clip_by_value_38/Minimumtraining/Adam/Const_76*
T0*
_output_shapes	
:
c
training/Adam/Sqrt_38Sqrttraining/Adam/clip_by_value_38*
_output_shapes	
:*
T0
\
training/Adam/add_115/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
r
training/Adam/add_115Addtraining/Adam/Sqrt_38training/Adam/add_115/y*
_output_shapes	
:*
T0
w
training/Adam/truediv_39RealDivtraining/Adam/mul_192training/Adam/add_115*
T0*
_output_shapes	
:
p
training/Adam/sub_115Subconv2d_6/bias/readtraining/Adam/truediv_39*
T0*
_output_shapes	
:
Ń
training/Adam/Assign_111Assigntraining/Adam/Variable_37training/Adam/add_113*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_37*
validate_shape(*
_output_shapes	
:
Ń
training/Adam/Assign_112Assigntraining/Adam/Variable_83training/Adam/add_114*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_83*
validate_shape(*
_output_shapes	
:
š
training/Adam/Assign_113Assignconv2d_6/biastraining/Adam/sub_115*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@conv2d_6/bias*
validate_shape(

training/Adam/mul_193MulAdam/beta_1/readtraining/Adam/Variable_38/read*
T0*'
_output_shapes
:@
\
training/Adam/sub_116/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_116Subtraining/Adam/sub_116/xAdam/beta_1/read*
T0*
_output_shapes
: 
­
training/Adam/mul_194Multraining/Adam/sub_116Ftraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
|
training/Adam/add_116Addtraining/Adam/mul_193training/Adam/mul_194*'
_output_shapes
:@*
T0

training/Adam/mul_195MulAdam/beta_2/readtraining/Adam/Variable_84/read*
T0*'
_output_shapes
:@
\
training/Adam/sub_117/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_117Subtraining/Adam/sub_117/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_38SquareFtraining/Adam/gradients/conv2d_7/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
~
training/Adam/mul_196Multraining/Adam/sub_117training/Adam/Square_38*'
_output_shapes
:@*
T0
|
training/Adam/add_117Addtraining/Adam/mul_195training/Adam/mul_196*
T0*'
_output_shapes
:@
z
training/Adam/mul_197Multraining/Adam/mul_2training/Adam/add_116*
T0*'
_output_shapes
:@
[
training/Adam/Const_78Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_79Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_39/MinimumMinimumtraining/Adam/add_117training/Adam/Const_79*'
_output_shapes
:@*
T0

training/Adam/clip_by_value_39Maximum&training/Adam/clip_by_value_39/Minimumtraining/Adam/Const_78*
T0*'
_output_shapes
:@
o
training/Adam/Sqrt_39Sqrttraining/Adam/clip_by_value_39*
T0*'
_output_shapes
:@
\
training/Adam/add_118/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
~
training/Adam/add_118Addtraining/Adam/Sqrt_39training/Adam/add_118/y*
T0*'
_output_shapes
:@

training/Adam/truediv_40RealDivtraining/Adam/mul_197training/Adam/add_118*'
_output_shapes
:@*
T0
~
training/Adam/sub_118Subconv2d_7/kernel/readtraining/Adam/truediv_40*
T0*'
_output_shapes
:@
Ý
training/Adam/Assign_114Assigntraining/Adam/Variable_38training/Adam/add_116*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_38
Ý
training/Adam/Assign_115Assigntraining/Adam/Variable_84training/Adam/add_117*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_84*
validate_shape(*'
_output_shapes
:@
É
training/Adam/Assign_116Assignconv2d_7/kerneltraining/Adam/sub_118*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@conv2d_7/kernel
s
training/Adam/mul_198MulAdam/beta_1/readtraining/Adam/Variable_39/read*
_output_shapes
:@*
T0
\
training/Adam/sub_119/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_119Subtraining/Adam/sub_119/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_199Multraining/Adam/sub_1199training/Adam/gradients/conv2d_7/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
o
training/Adam/add_119Addtraining/Adam/mul_198training/Adam/mul_199*
T0*
_output_shapes
:@
s
training/Adam/mul_200MulAdam/beta_2/readtraining/Adam/Variable_85/read*
_output_shapes
:@*
T0
\
training/Adam/sub_120/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_120Subtraining/Adam/sub_120/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_39Square9training/Adam/gradients/conv2d_7/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
q
training/Adam/mul_201Multraining/Adam/sub_120training/Adam/Square_39*
T0*
_output_shapes
:@
o
training/Adam/add_120Addtraining/Adam/mul_200training/Adam/mul_201*
_output_shapes
:@*
T0
m
training/Adam/mul_202Multraining/Adam/mul_2training/Adam/add_119*
_output_shapes
:@*
T0
[
training/Adam/Const_80Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_81Const*
dtype0*
_output_shapes
: *
valueB
 *  

&training/Adam/clip_by_value_40/MinimumMinimumtraining/Adam/add_120training/Adam/Const_81*
T0*
_output_shapes
:@

training/Adam/clip_by_value_40Maximum&training/Adam/clip_by_value_40/Minimumtraining/Adam/Const_80*
T0*
_output_shapes
:@
b
training/Adam/Sqrt_40Sqrttraining/Adam/clip_by_value_40*
_output_shapes
:@*
T0
\
training/Adam/add_121/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
q
training/Adam/add_121Addtraining/Adam/Sqrt_40training/Adam/add_121/y*
T0*
_output_shapes
:@
v
training/Adam/truediv_41RealDivtraining/Adam/mul_202training/Adam/add_121*
T0*
_output_shapes
:@
o
training/Adam/sub_121Subconv2d_7/bias/readtraining/Adam/truediv_41*
T0*
_output_shapes
:@
Đ
training/Adam/Assign_117Assigntraining/Adam/Variable_39training/Adam/add_119*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_39
Đ
training/Adam/Assign_118Assigntraining/Adam/Variable_85training/Adam/add_120*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_85*
validate_shape(*
_output_shapes
:@
¸
training/Adam/Assign_119Assignconv2d_7/biastraining/Adam/sub_121*
use_locking(*
T0* 
_class
loc:@conv2d_7/bias*
validate_shape(*
_output_shapes
:@

training/Adam/mul_203MulAdam/beta_1/readtraining/Adam/Variable_40/read*
T0*'
_output_shapes
:@
\
training/Adam/sub_122/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_122Subtraining/Adam/sub_122/xAdam/beta_1/read*
_output_shapes
: *
T0
­
training/Adam/mul_204Multraining/Adam/sub_122Ftraining/Adam/gradients/conv2d_8/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
|
training/Adam/add_122Addtraining/Adam/mul_203training/Adam/mul_204*'
_output_shapes
:@*
T0

training/Adam/mul_205MulAdam/beta_2/readtraining/Adam/Variable_86/read*
T0*'
_output_shapes
:@
\
training/Adam/sub_123/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_123Subtraining/Adam/sub_123/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_40SquareFtraining/Adam/gradients/conv2d_8/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
~
training/Adam/mul_206Multraining/Adam/sub_123training/Adam/Square_40*'
_output_shapes
:@*
T0
|
training/Adam/add_123Addtraining/Adam/mul_205training/Adam/mul_206*'
_output_shapes
:@*
T0
z
training/Adam/mul_207Multraining/Adam/mul_2training/Adam/add_122*
T0*'
_output_shapes
:@
[
training/Adam/Const_82Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_83Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_41/MinimumMinimumtraining/Adam/add_123training/Adam/Const_83*
T0*'
_output_shapes
:@

training/Adam/clip_by_value_41Maximum&training/Adam/clip_by_value_41/Minimumtraining/Adam/Const_82*
T0*'
_output_shapes
:@
o
training/Adam/Sqrt_41Sqrttraining/Adam/clip_by_value_41*
T0*'
_output_shapes
:@
\
training/Adam/add_124/yConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
~
training/Adam/add_124Addtraining/Adam/Sqrt_41training/Adam/add_124/y*
T0*'
_output_shapes
:@

training/Adam/truediv_42RealDivtraining/Adam/mul_207training/Adam/add_124*'
_output_shapes
:@*
T0
~
training/Adam/sub_124Subconv2d_8/kernel/readtraining/Adam/truediv_42*'
_output_shapes
:@*
T0
Ý
training/Adam/Assign_120Assigntraining/Adam/Variable_40training/Adam/add_122*
T0*,
_class"
 loc:@training/Adam/Variable_40*
validate_shape(*'
_output_shapes
:@*
use_locking(
Ý
training/Adam/Assign_121Assigntraining/Adam/Variable_86training/Adam/add_123*'
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_86*
validate_shape(
É
training/Adam/Assign_122Assignconv2d_8/kerneltraining/Adam/sub_124*
use_locking(*
T0*"
_class
loc:@conv2d_8/kernel*
validate_shape(*'
_output_shapes
:@
s
training/Adam/mul_208MulAdam/beta_1/readtraining/Adam/Variable_41/read*
_output_shapes
:@*
T0
\
training/Adam/sub_125/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_125Subtraining/Adam/sub_125/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_209Multraining/Adam/sub_1259training/Adam/gradients/conv2d_8/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
o
training/Adam/add_125Addtraining/Adam/mul_208training/Adam/mul_209*
T0*
_output_shapes
:@
s
training/Adam/mul_210MulAdam/beta_2/readtraining/Adam/Variable_87/read*
T0*
_output_shapes
:@
\
training/Adam/sub_126/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_126Subtraining/Adam/sub_126/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_41Square9training/Adam/gradients/conv2d_8/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
q
training/Adam/mul_211Multraining/Adam/sub_126training/Adam/Square_41*
T0*
_output_shapes
:@
o
training/Adam/add_126Addtraining/Adam/mul_210training/Adam/mul_211*
T0*
_output_shapes
:@
m
training/Adam/mul_212Multraining/Adam/mul_2training/Adam/add_125*
T0*
_output_shapes
:@
[
training/Adam/Const_84Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_85Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_42/MinimumMinimumtraining/Adam/add_126training/Adam/Const_85*
T0*
_output_shapes
:@

training/Adam/clip_by_value_42Maximum&training/Adam/clip_by_value_42/Minimumtraining/Adam/Const_84*
T0*
_output_shapes
:@
b
training/Adam/Sqrt_42Sqrttraining/Adam/clip_by_value_42*
T0*
_output_shapes
:@
\
training/Adam/add_127/yConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
q
training/Adam/add_127Addtraining/Adam/Sqrt_42training/Adam/add_127/y*
T0*
_output_shapes
:@
v
training/Adam/truediv_43RealDivtraining/Adam/mul_212training/Adam/add_127*
_output_shapes
:@*
T0
o
training/Adam/sub_127Subconv2d_8/bias/readtraining/Adam/truediv_43*
_output_shapes
:@*
T0
Đ
training/Adam/Assign_123Assigntraining/Adam/Variable_41training/Adam/add_125*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_41*
validate_shape(*
_output_shapes
:@
Đ
training/Adam/Assign_124Assigntraining/Adam/Variable_87training/Adam/add_126*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_87*
validate_shape(*
_output_shapes
:@
¸
training/Adam/Assign_125Assignconv2d_8/biastraining/Adam/sub_127*
use_locking(*
T0* 
_class
loc:@conv2d_8/bias*
validate_shape(*
_output_shapes
:@

training/Adam/mul_213MulAdam/beta_1/readtraining/Adam/Variable_42/read*
T0*&
_output_shapes
:@@
\
training/Adam/sub_128/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
h
training/Adam/sub_128Subtraining/Adam/sub_128/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ź
training/Adam/mul_214Multraining/Adam/sub_128Ftraining/Adam/gradients/conv2d_9/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
{
training/Adam/add_128Addtraining/Adam/mul_213training/Adam/mul_214*
T0*&
_output_shapes
:@@

training/Adam/mul_215MulAdam/beta_2/readtraining/Adam/Variable_88/read*
T0*&
_output_shapes
:@@
\
training/Adam/sub_129/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
h
training/Adam/sub_129Subtraining/Adam/sub_129/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_42SquareFtraining/Adam/gradients/conv2d_9/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@@
}
training/Adam/mul_216Multraining/Adam/sub_129training/Adam/Square_42*
T0*&
_output_shapes
:@@
{
training/Adam/add_129Addtraining/Adam/mul_215training/Adam/mul_216*
T0*&
_output_shapes
:@@
y
training/Adam/mul_217Multraining/Adam/mul_2training/Adam/add_128*&
_output_shapes
:@@*
T0
[
training/Adam/Const_86Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_87Const*
_output_shapes
: *
valueB
 *  *
dtype0

&training/Adam/clip_by_value_43/MinimumMinimumtraining/Adam/add_129training/Adam/Const_87*
T0*&
_output_shapes
:@@

training/Adam/clip_by_value_43Maximum&training/Adam/clip_by_value_43/Minimumtraining/Adam/Const_86*
T0*&
_output_shapes
:@@
n
training/Adam/Sqrt_43Sqrttraining/Adam/clip_by_value_43*
T0*&
_output_shapes
:@@
\
training/Adam/add_130/yConst*
_output_shapes
: *
valueB
 *żÖ3*
dtype0
}
training/Adam/add_130Addtraining/Adam/Sqrt_43training/Adam/add_130/y*
T0*&
_output_shapes
:@@

training/Adam/truediv_44RealDivtraining/Adam/mul_217training/Adam/add_130*&
_output_shapes
:@@*
T0
}
training/Adam/sub_130Subconv2d_9/kernel/readtraining/Adam/truediv_44*
T0*&
_output_shapes
:@@
Ü
training/Adam/Assign_126Assigntraining/Adam/Variable_42training/Adam/add_128*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_42*
validate_shape(*&
_output_shapes
:@@
Ü
training/Adam/Assign_127Assigntraining/Adam/Variable_88training/Adam/add_129*
T0*,
_class"
 loc:@training/Adam/Variable_88*
validate_shape(*&
_output_shapes
:@@*
use_locking(
Č
training/Adam/Assign_128Assignconv2d_9/kerneltraining/Adam/sub_130*
use_locking(*
T0*"
_class
loc:@conv2d_9/kernel*
validate_shape(*&
_output_shapes
:@@
s
training/Adam/mul_218MulAdam/beta_1/readtraining/Adam/Variable_43/read*
_output_shapes
:@*
T0
\
training/Adam/sub_131/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_131Subtraining/Adam/sub_131/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_219Multraining/Adam/sub_1319training/Adam/gradients/conv2d_9/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
o
training/Adam/add_131Addtraining/Adam/mul_218training/Adam/mul_219*
T0*
_output_shapes
:@
s
training/Adam/mul_220MulAdam/beta_2/readtraining/Adam/Variable_89/read*
T0*
_output_shapes
:@
\
training/Adam/sub_132/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
h
training/Adam/sub_132Subtraining/Adam/sub_132/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_43Square9training/Adam/gradients/conv2d_9/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
q
training/Adam/mul_221Multraining/Adam/sub_132training/Adam/Square_43*
_output_shapes
:@*
T0
o
training/Adam/add_132Addtraining/Adam/mul_220training/Adam/mul_221*
_output_shapes
:@*
T0
m
training/Adam/mul_222Multraining/Adam/mul_2training/Adam/add_131*
T0*
_output_shapes
:@
[
training/Adam/Const_88Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_89Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_44/MinimumMinimumtraining/Adam/add_132training/Adam/Const_89*
T0*
_output_shapes
:@

training/Adam/clip_by_value_44Maximum&training/Adam/clip_by_value_44/Minimumtraining/Adam/Const_88*
_output_shapes
:@*
T0
b
training/Adam/Sqrt_44Sqrttraining/Adam/clip_by_value_44*
T0*
_output_shapes
:@
\
training/Adam/add_133/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
q
training/Adam/add_133Addtraining/Adam/Sqrt_44training/Adam/add_133/y*
_output_shapes
:@*
T0
v
training/Adam/truediv_45RealDivtraining/Adam/mul_222training/Adam/add_133*
_output_shapes
:@*
T0
o
training/Adam/sub_133Subconv2d_9/bias/readtraining/Adam/truediv_45*
_output_shapes
:@*
T0
Đ
training/Adam/Assign_129Assigntraining/Adam/Variable_43training/Adam/add_131*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_43*
validate_shape(*
_output_shapes
:@
Đ
training/Adam/Assign_130Assigntraining/Adam/Variable_89training/Adam/add_132*,
_class"
 loc:@training/Adam/Variable_89*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
¸
training/Adam/Assign_131Assignconv2d_9/biastraining/Adam/sub_133*
use_locking(*
T0* 
_class
loc:@conv2d_9/bias*
validate_shape(*
_output_shapes
:@

training/Adam/mul_223MulAdam/beta_1/readtraining/Adam/Variable_44/read*
T0*&
_output_shapes
:@
\
training/Adam/sub_134/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
h
training/Adam/sub_134Subtraining/Adam/sub_134/xAdam/beta_1/read*
T0*
_output_shapes
: 
­
training/Adam/mul_224Multraining/Adam/sub_134Gtraining/Adam/gradients/conv2d_10/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
{
training/Adam/add_134Addtraining/Adam/mul_223training/Adam/mul_224*
T0*&
_output_shapes
:@

training/Adam/mul_225MulAdam/beta_2/readtraining/Adam/Variable_90/read*
T0*&
_output_shapes
:@
\
training/Adam/sub_135/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training/Adam/sub_135Subtraining/Adam/sub_135/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_44SquareGtraining/Adam/gradients/conv2d_10/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@*
T0
}
training/Adam/mul_226Multraining/Adam/sub_135training/Adam/Square_44*
T0*&
_output_shapes
:@
{
training/Adam/add_135Addtraining/Adam/mul_225training/Adam/mul_226*
T0*&
_output_shapes
:@
y
training/Adam/mul_227Multraining/Adam/mul_2training/Adam/add_134*&
_output_shapes
:@*
T0
[
training/Adam/Const_90Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_91Const*
_output_shapes
: *
valueB
 *  *
dtype0

&training/Adam/clip_by_value_45/MinimumMinimumtraining/Adam/add_135training/Adam/Const_91*
T0*&
_output_shapes
:@

training/Adam/clip_by_value_45Maximum&training/Adam/clip_by_value_45/Minimumtraining/Adam/Const_90*
T0*&
_output_shapes
:@
n
training/Adam/Sqrt_45Sqrttraining/Adam/clip_by_value_45*&
_output_shapes
:@*
T0
\
training/Adam/add_136/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
}
training/Adam/add_136Addtraining/Adam/Sqrt_45training/Adam/add_136/y*
T0*&
_output_shapes
:@

training/Adam/truediv_46RealDivtraining/Adam/mul_227training/Adam/add_136*
T0*&
_output_shapes
:@
~
training/Adam/sub_136Subconv2d_10/kernel/readtraining/Adam/truediv_46*
T0*&
_output_shapes
:@
Ü
training/Adam/Assign_132Assigntraining/Adam/Variable_44training/Adam/add_134*
T0*,
_class"
 loc:@training/Adam/Variable_44*
validate_shape(*&
_output_shapes
:@*
use_locking(
Ü
training/Adam/Assign_133Assigntraining/Adam/Variable_90training/Adam/add_135*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_90
Ę
training/Adam/Assign_134Assignconv2d_10/kerneltraining/Adam/sub_136*&
_output_shapes
:@*
use_locking(*
T0*#
_class
loc:@conv2d_10/kernel*
validate_shape(
s
training/Adam/mul_228MulAdam/beta_1/readtraining/Adam/Variable_45/read*
_output_shapes
:*
T0
\
training/Adam/sub_137/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
h
training/Adam/sub_137Subtraining/Adam/sub_137/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_229Multraining/Adam/sub_137:training/Adam/gradients/conv2d_10/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
o
training/Adam/add_137Addtraining/Adam/mul_228training/Adam/mul_229*
_output_shapes
:*
T0
s
training/Adam/mul_230MulAdam/beta_2/readtraining/Adam/Variable_91/read*
T0*
_output_shapes
:
\
training/Adam/sub_138/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
h
training/Adam/sub_138Subtraining/Adam/sub_138/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_45Square:training/Adam/gradients/conv2d_10/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
q
training/Adam/mul_231Multraining/Adam/sub_138training/Adam/Square_45*
T0*
_output_shapes
:
o
training/Adam/add_138Addtraining/Adam/mul_230training/Adam/mul_231*
T0*
_output_shapes
:
m
training/Adam/mul_232Multraining/Adam/mul_2training/Adam/add_137*
T0*
_output_shapes
:
[
training/Adam/Const_92Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_93Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/Adam/clip_by_value_46/MinimumMinimumtraining/Adam/add_138training/Adam/Const_93*
T0*
_output_shapes
:

training/Adam/clip_by_value_46Maximum&training/Adam/clip_by_value_46/Minimumtraining/Adam/Const_92*
T0*
_output_shapes
:
b
training/Adam/Sqrt_46Sqrttraining/Adam/clip_by_value_46*
T0*
_output_shapes
:
\
training/Adam/add_139/yConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
q
training/Adam/add_139Addtraining/Adam/Sqrt_46training/Adam/add_139/y*
T0*
_output_shapes
:
v
training/Adam/truediv_47RealDivtraining/Adam/mul_232training/Adam/add_139*
T0*
_output_shapes
:
p
training/Adam/sub_139Subconv2d_10/bias/readtraining/Adam/truediv_47*
_output_shapes
:*
T0
Đ
training/Adam/Assign_135Assigntraining/Adam/Variable_45training/Adam/add_137*
T0*,
_class"
 loc:@training/Adam/Variable_45*
validate_shape(*
_output_shapes
:*
use_locking(
Đ
training/Adam/Assign_136Assigntraining/Adam/Variable_91training/Adam/add_138*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_91*
validate_shape(*
_output_shapes
:
ş
training/Adam/Assign_137Assignconv2d_10/biastraining/Adam/sub_139*
T0*!
_class
loc:@conv2d_10/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ţ
training/group_depsNoOp	^loss/mul^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_100^training/Adam/Assign_101^training/Adam/Assign_102^training/Adam/Assign_103^training/Adam/Assign_104^training/Adam/Assign_105^training/Adam/Assign_106^training/Adam/Assign_107^training/Adam/Assign_108^training/Adam/Assign_109^training/Adam/Assign_11^training/Adam/Assign_110^training/Adam/Assign_111^training/Adam/Assign_112^training/Adam/Assign_113^training/Adam/Assign_114^training/Adam/Assign_115^training/Adam/Assign_116^training/Adam/Assign_117^training/Adam/Assign_118^training/Adam/Assign_119^training/Adam/Assign_12^training/Adam/Assign_120^training/Adam/Assign_121^training/Adam/Assign_122^training/Adam/Assign_123^training/Adam/Assign_124^training/Adam/Assign_125^training/Adam/Assign_126^training/Adam/Assign_127^training/Adam/Assign_128^training/Adam/Assign_129^training/Adam/Assign_13^training/Adam/Assign_130^training/Adam/Assign_131^training/Adam/Assign_132^training/Adam/Assign_133^training/Adam/Assign_134^training/Adam/Assign_135^training/Adam/Assign_136^training/Adam/Assign_137^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_24^training/Adam/Assign_25^training/Adam/Assign_26^training/Adam/Assign_27^training/Adam/Assign_28^training/Adam/Assign_29^training/Adam/Assign_3^training/Adam/Assign_30^training/Adam/Assign_31^training/Adam/Assign_32^training/Adam/Assign_33^training/Adam/Assign_34^training/Adam/Assign_35^training/Adam/Assign_36^training/Adam/Assign_37^training/Adam/Assign_38^training/Adam/Assign_39^training/Adam/Assign_4^training/Adam/Assign_40^training/Adam/Assign_41^training/Adam/Assign_42^training/Adam/Assign_43^training/Adam/Assign_44^training/Adam/Assign_45^training/Adam/Assign_46^training/Adam/Assign_47^training/Adam/Assign_48^training/Adam/Assign_49^training/Adam/Assign_5^training/Adam/Assign_50^training/Adam/Assign_51^training/Adam/Assign_52^training/Adam/Assign_53^training/Adam/Assign_54^training/Adam/Assign_55^training/Adam/Assign_56^training/Adam/Assign_57^training/Adam/Assign_58^training/Adam/Assign_59^training/Adam/Assign_6^training/Adam/Assign_60^training/Adam/Assign_61^training/Adam/Assign_62^training/Adam/Assign_63^training/Adam/Assign_64^training/Adam/Assign_65^training/Adam/Assign_66^training/Adam/Assign_67^training/Adam/Assign_68^training/Adam/Assign_69^training/Adam/Assign_7^training/Adam/Assign_70^training/Adam/Assign_71^training/Adam/Assign_72^training/Adam/Assign_73^training/Adam/Assign_74^training/Adam/Assign_75^training/Adam/Assign_76^training/Adam/Assign_77^training/Adam/Assign_78^training/Adam/Assign_79^training/Adam/Assign_8^training/Adam/Assign_80^training/Adam/Assign_81^training/Adam/Assign_82^training/Adam/Assign_83^training/Adam/Assign_84^training/Adam/Assign_85^training/Adam/Assign_86^training/Adam/Assign_87^training/Adam/Assign_88^training/Adam/Assign_89^training/Adam/Assign_9^training/Adam/Assign_90^training/Adam/Assign_91^training/Adam/Assign_92^training/Adam/Assign_93^training/Adam/Assign_94^training/Adam/Assign_95^training/Adam/Assign_96^training/Adam/Assign_97^training/Adam/Assign_98^training/Adam/Assign_99


group_depsNoOp	^loss/mul

IsVariableInitializedIsVariableInitializedconv1a/kernel* 
_class
loc:@conv1a/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitializedconv1a/bias*
_class
loc:@conv1a/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_2IsVariableInitializedconv1b/kernel* 
_class
loc:@conv1b/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_3IsVariableInitializedconv1b/bias*
_class
loc:@conv1b/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializedconv2a/kernel* 
_class
loc:@conv2a/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializedconv2a/bias*
dtype0*
_output_shapes
: *
_class
loc:@conv2a/bias

IsVariableInitialized_6IsVariableInitializedconv2b/kernel* 
_class
loc:@conv2b/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_7IsVariableInitializedconv2b/bias*
_class
loc:@conv2b/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializedconv3a/kernel* 
_class
loc:@conv3a/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_9IsVariableInitializedconv3a/bias*
_class
loc:@conv3a/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitializedconv3b/kernel* 
_class
loc:@conv3b/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_11IsVariableInitializedconv3b/bias*
_output_shapes
: *
_class
loc:@conv3b/bias*
dtype0

IsVariableInitialized_12IsVariableInitializedconv4a/kernel* 
_class
loc:@conv4a/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_13IsVariableInitializedconv4a/bias*
_class
loc:@conv4a/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_14IsVariableInitializedconv4b/kernel* 
_class
loc:@conv4b/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_15IsVariableInitializedconv4b/bias*
_class
loc:@conv4b/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_16IsVariableInitializedconv5a/kernel*
dtype0*
_output_shapes
: * 
_class
loc:@conv5a/kernel

IsVariableInitialized_17IsVariableInitializedconv5a/bias*
_class
loc:@conv5a/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_18IsVariableInitializedconv5b/kernel* 
_class
loc:@conv5b/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_19IsVariableInitializedconv5b/bias*
_class
loc:@conv5b/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_20IsVariableInitializedconv6/kernel*
_class
loc:@conv6/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitialized
conv6/bias*
dtype0*
_output_shapes
: *
_class
loc:@conv6/bias

IsVariableInitialized_22IsVariableInitializedconv7a/kernel* 
_class
loc:@conv7a/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_23IsVariableInitializedconv7a/bias*
_class
loc:@conv7a/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_24IsVariableInitializedconv7b/kernel* 
_class
loc:@conv7b/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_25IsVariableInitializedconv7b/bias*
_class
loc:@conv7b/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_26IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_27IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_28IsVariableInitializedconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_29IsVariableInitializedconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_30IsVariableInitializedconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_31IsVariableInitializedconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_32IsVariableInitializedconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_33IsVariableInitializedconv2d_4/bias*
_output_shapes
: * 
_class
loc:@conv2d_4/bias*
dtype0

IsVariableInitialized_34IsVariableInitializedconv2d_5/kernel*"
_class
loc:@conv2d_5/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_35IsVariableInitializedconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_36IsVariableInitializedconv2d_6/kernel*"
_class
loc:@conv2d_6/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_37IsVariableInitializedconv2d_6/bias* 
_class
loc:@conv2d_6/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_38IsVariableInitializedconv2d_7/kernel*"
_class
loc:@conv2d_7/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_39IsVariableInitializedconv2d_7/bias*
_output_shapes
: * 
_class
loc:@conv2d_7/bias*
dtype0

IsVariableInitialized_40IsVariableInitializedconv2d_8/kernel*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_8/kernel

IsVariableInitialized_41IsVariableInitializedconv2d_8/bias* 
_class
loc:@conv2d_8/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_42IsVariableInitializedconv2d_9/kernel*"
_class
loc:@conv2d_9/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_43IsVariableInitializedconv2d_9/bias* 
_class
loc:@conv2d_9/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_44IsVariableInitializedconv2d_10/kernel*
_output_shapes
: *#
_class
loc:@conv2d_10/kernel*
dtype0

IsVariableInitialized_45IsVariableInitializedconv2d_10/bias*!
_class
loc:@conv2d_10/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_46IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
{
IsVariableInitialized_47IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 

IsVariableInitialized_48IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_49IsVariableInitializedAdam/beta_2*
_output_shapes
: *
_class
loc:@Adam/beta_2*
dtype0

IsVariableInitialized_50IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 

IsVariableInitialized_51IsVariableInitializedtraining/Adam/Variable*
_output_shapes
: *)
_class
loc:@training/Adam/Variable*
dtype0

IsVariableInitialized_52IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_53IsVariableInitializedtraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_54IsVariableInitializedtraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0*
_output_shapes
: 

IsVariableInitialized_55IsVariableInitializedtraining/Adam/Variable_4*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_4

IsVariableInitialized_56IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
: 

IsVariableInitialized_57IsVariableInitializedtraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes
: 

IsVariableInitialized_58IsVariableInitializedtraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0*
_output_shapes
: 

IsVariableInitialized_59IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes
: 

IsVariableInitialized_60IsVariableInitializedtraining/Adam/Variable_9*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_9

IsVariableInitialized_61IsVariableInitializedtraining/Adam/Variable_10*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_10*
dtype0

IsVariableInitialized_62IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
: 

IsVariableInitialized_63IsVariableInitializedtraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
: 

IsVariableInitialized_64IsVariableInitializedtraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
dtype0*
_output_shapes
: 

IsVariableInitialized_65IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes
: 

IsVariableInitialized_66IsVariableInitializedtraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
dtype0*
_output_shapes
: 

IsVariableInitialized_67IsVariableInitializedtraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0*
_output_shapes
: 

IsVariableInitialized_68IsVariableInitializedtraining/Adam/Variable_17*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_17*
dtype0

IsVariableInitialized_69IsVariableInitializedtraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
dtype0*
_output_shapes
: 

IsVariableInitialized_70IsVariableInitializedtraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
dtype0*
_output_shapes
: 

IsVariableInitialized_71IsVariableInitializedtraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0*
_output_shapes
: 

IsVariableInitialized_72IsVariableInitializedtraining/Adam/Variable_21*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_21*
dtype0

IsVariableInitialized_73IsVariableInitializedtraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
dtype0*
_output_shapes
: 

IsVariableInitialized_74IsVariableInitializedtraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
dtype0*
_output_shapes
: 

IsVariableInitialized_75IsVariableInitializedtraining/Adam/Variable_24*,
_class"
 loc:@training/Adam/Variable_24*
dtype0*
_output_shapes
: 

IsVariableInitialized_76IsVariableInitializedtraining/Adam/Variable_25*,
_class"
 loc:@training/Adam/Variable_25*
dtype0*
_output_shapes
: 

IsVariableInitialized_77IsVariableInitializedtraining/Adam/Variable_26*,
_class"
 loc:@training/Adam/Variable_26*
dtype0*
_output_shapes
: 

IsVariableInitialized_78IsVariableInitializedtraining/Adam/Variable_27*,
_class"
 loc:@training/Adam/Variable_27*
dtype0*
_output_shapes
: 

IsVariableInitialized_79IsVariableInitializedtraining/Adam/Variable_28*,
_class"
 loc:@training/Adam/Variable_28*
dtype0*
_output_shapes
: 

IsVariableInitialized_80IsVariableInitializedtraining/Adam/Variable_29*,
_class"
 loc:@training/Adam/Variable_29*
dtype0*
_output_shapes
: 

IsVariableInitialized_81IsVariableInitializedtraining/Adam/Variable_30*,
_class"
 loc:@training/Adam/Variable_30*
dtype0*
_output_shapes
: 

IsVariableInitialized_82IsVariableInitializedtraining/Adam/Variable_31*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_31

IsVariableInitialized_83IsVariableInitializedtraining/Adam/Variable_32*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_32*
dtype0

IsVariableInitialized_84IsVariableInitializedtraining/Adam/Variable_33*,
_class"
 loc:@training/Adam/Variable_33*
dtype0*
_output_shapes
: 

IsVariableInitialized_85IsVariableInitializedtraining/Adam/Variable_34*,
_class"
 loc:@training/Adam/Variable_34*
dtype0*
_output_shapes
: 

IsVariableInitialized_86IsVariableInitializedtraining/Adam/Variable_35*,
_class"
 loc:@training/Adam/Variable_35*
dtype0*
_output_shapes
: 

IsVariableInitialized_87IsVariableInitializedtraining/Adam/Variable_36*,
_class"
 loc:@training/Adam/Variable_36*
dtype0*
_output_shapes
: 

IsVariableInitialized_88IsVariableInitializedtraining/Adam/Variable_37*,
_class"
 loc:@training/Adam/Variable_37*
dtype0*
_output_shapes
: 

IsVariableInitialized_89IsVariableInitializedtraining/Adam/Variable_38*,
_class"
 loc:@training/Adam/Variable_38*
dtype0*
_output_shapes
: 

IsVariableInitialized_90IsVariableInitializedtraining/Adam/Variable_39*,
_class"
 loc:@training/Adam/Variable_39*
dtype0*
_output_shapes
: 

IsVariableInitialized_91IsVariableInitializedtraining/Adam/Variable_40*,
_class"
 loc:@training/Adam/Variable_40*
dtype0*
_output_shapes
: 

IsVariableInitialized_92IsVariableInitializedtraining/Adam/Variable_41*,
_class"
 loc:@training/Adam/Variable_41*
dtype0*
_output_shapes
: 

IsVariableInitialized_93IsVariableInitializedtraining/Adam/Variable_42*,
_class"
 loc:@training/Adam/Variable_42*
dtype0*
_output_shapes
: 

IsVariableInitialized_94IsVariableInitializedtraining/Adam/Variable_43*,
_class"
 loc:@training/Adam/Variable_43*
dtype0*
_output_shapes
: 

IsVariableInitialized_95IsVariableInitializedtraining/Adam/Variable_44*,
_class"
 loc:@training/Adam/Variable_44*
dtype0*
_output_shapes
: 

IsVariableInitialized_96IsVariableInitializedtraining/Adam/Variable_45*,
_class"
 loc:@training/Adam/Variable_45*
dtype0*
_output_shapes
: 

IsVariableInitialized_97IsVariableInitializedtraining/Adam/Variable_46*,
_class"
 loc:@training/Adam/Variable_46*
dtype0*
_output_shapes
: 

IsVariableInitialized_98IsVariableInitializedtraining/Adam/Variable_47*,
_class"
 loc:@training/Adam/Variable_47*
dtype0*
_output_shapes
: 

IsVariableInitialized_99IsVariableInitializedtraining/Adam/Variable_48*,
_class"
 loc:@training/Adam/Variable_48*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_100IsVariableInitializedtraining/Adam/Variable_49*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_49*
dtype0
 
IsVariableInitialized_101IsVariableInitializedtraining/Adam/Variable_50*,
_class"
 loc:@training/Adam/Variable_50*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_102IsVariableInitializedtraining/Adam/Variable_51*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_51
 
IsVariableInitialized_103IsVariableInitializedtraining/Adam/Variable_52*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_52*
dtype0
 
IsVariableInitialized_104IsVariableInitializedtraining/Adam/Variable_53*,
_class"
 loc:@training/Adam/Variable_53*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_105IsVariableInitializedtraining/Adam/Variable_54*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_54*
dtype0
 
IsVariableInitialized_106IsVariableInitializedtraining/Adam/Variable_55*,
_class"
 loc:@training/Adam/Variable_55*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_107IsVariableInitializedtraining/Adam/Variable_56*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_56
 
IsVariableInitialized_108IsVariableInitializedtraining/Adam/Variable_57*,
_class"
 loc:@training/Adam/Variable_57*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_109IsVariableInitializedtraining/Adam/Variable_58*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_58*
dtype0
 
IsVariableInitialized_110IsVariableInitializedtraining/Adam/Variable_59*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_59*
dtype0
 
IsVariableInitialized_111IsVariableInitializedtraining/Adam/Variable_60*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_60*
dtype0
 
IsVariableInitialized_112IsVariableInitializedtraining/Adam/Variable_61*,
_class"
 loc:@training/Adam/Variable_61*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_113IsVariableInitializedtraining/Adam/Variable_62*,
_class"
 loc:@training/Adam/Variable_62*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_114IsVariableInitializedtraining/Adam/Variable_63*,
_class"
 loc:@training/Adam/Variable_63*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_115IsVariableInitializedtraining/Adam/Variable_64*,
_class"
 loc:@training/Adam/Variable_64*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_116IsVariableInitializedtraining/Adam/Variable_65*,
_class"
 loc:@training/Adam/Variable_65*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_117IsVariableInitializedtraining/Adam/Variable_66*,
_class"
 loc:@training/Adam/Variable_66*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_118IsVariableInitializedtraining/Adam/Variable_67*,
_class"
 loc:@training/Adam/Variable_67*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_119IsVariableInitializedtraining/Adam/Variable_68*,
_class"
 loc:@training/Adam/Variable_68*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_120IsVariableInitializedtraining/Adam/Variable_69*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_69*
dtype0
 
IsVariableInitialized_121IsVariableInitializedtraining/Adam/Variable_70*,
_class"
 loc:@training/Adam/Variable_70*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_122IsVariableInitializedtraining/Adam/Variable_71*,
_class"
 loc:@training/Adam/Variable_71*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_123IsVariableInitializedtraining/Adam/Variable_72*,
_class"
 loc:@training/Adam/Variable_72*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_124IsVariableInitializedtraining/Adam/Variable_73*,
_class"
 loc:@training/Adam/Variable_73*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_125IsVariableInitializedtraining/Adam/Variable_74*,
_class"
 loc:@training/Adam/Variable_74*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_126IsVariableInitializedtraining/Adam/Variable_75*,
_class"
 loc:@training/Adam/Variable_75*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_127IsVariableInitializedtraining/Adam/Variable_76*,
_class"
 loc:@training/Adam/Variable_76*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_128IsVariableInitializedtraining/Adam/Variable_77*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_77
 
IsVariableInitialized_129IsVariableInitializedtraining/Adam/Variable_78*,
_class"
 loc:@training/Adam/Variable_78*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_130IsVariableInitializedtraining/Adam/Variable_79*,
_class"
 loc:@training/Adam/Variable_79*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_131IsVariableInitializedtraining/Adam/Variable_80*,
_class"
 loc:@training/Adam/Variable_80*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_132IsVariableInitializedtraining/Adam/Variable_81*,
_class"
 loc:@training/Adam/Variable_81*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_133IsVariableInitializedtraining/Adam/Variable_82*,
_class"
 loc:@training/Adam/Variable_82*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_134IsVariableInitializedtraining/Adam/Variable_83*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_83
 
IsVariableInitialized_135IsVariableInitializedtraining/Adam/Variable_84*,
_class"
 loc:@training/Adam/Variable_84*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_136IsVariableInitializedtraining/Adam/Variable_85*,
_class"
 loc:@training/Adam/Variable_85*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_137IsVariableInitializedtraining/Adam/Variable_86*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_86
 
IsVariableInitialized_138IsVariableInitializedtraining/Adam/Variable_87*,
_class"
 loc:@training/Adam/Variable_87*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_139IsVariableInitializedtraining/Adam/Variable_88*,
_class"
 loc:@training/Adam/Variable_88*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_140IsVariableInitializedtraining/Adam/Variable_89*,
_class"
 loc:@training/Adam/Variable_89*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_141IsVariableInitializedtraining/Adam/Variable_90*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_90
 
IsVariableInitialized_142IsVariableInitializedtraining/Adam/Variable_91*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_91
 
IsVariableInitialized_143IsVariableInitializedtraining/Adam/Variable_92*,
_class"
 loc:@training/Adam/Variable_92*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_144IsVariableInitializedtraining/Adam/Variable_93*,
_class"
 loc:@training/Adam/Variable_93*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_145IsVariableInitializedtraining/Adam/Variable_94*,
_class"
 loc:@training/Adam/Variable_94*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_146IsVariableInitializedtraining/Adam/Variable_95*,
_class"
 loc:@training/Adam/Variable_95*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_147IsVariableInitializedtraining/Adam/Variable_96*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_96*
dtype0
 
IsVariableInitialized_148IsVariableInitializedtraining/Adam/Variable_97*,
_class"
 loc:@training/Adam/Variable_97*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_149IsVariableInitializedtraining/Adam/Variable_98*,
_class"
 loc:@training/Adam/Variable_98*
dtype0*
_output_shapes
: 
 
IsVariableInitialized_150IsVariableInitializedtraining/Adam/Variable_99*,
_class"
 loc:@training/Adam/Variable_99*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_151IsVariableInitializedtraining/Adam/Variable_100*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_100*
dtype0
˘
IsVariableInitialized_152IsVariableInitializedtraining/Adam/Variable_101*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_101*
dtype0
˘
IsVariableInitialized_153IsVariableInitializedtraining/Adam/Variable_102*-
_class#
!loc:@training/Adam/Variable_102*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_154IsVariableInitializedtraining/Adam/Variable_103*-
_class#
!loc:@training/Adam/Variable_103*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_155IsVariableInitializedtraining/Adam/Variable_104*-
_class#
!loc:@training/Adam/Variable_104*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_156IsVariableInitializedtraining/Adam/Variable_105*
dtype0*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_105
˘
IsVariableInitialized_157IsVariableInitializedtraining/Adam/Variable_106*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_106*
dtype0
˘
IsVariableInitialized_158IsVariableInitializedtraining/Adam/Variable_107*-
_class#
!loc:@training/Adam/Variable_107*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_159IsVariableInitializedtraining/Adam/Variable_108*
dtype0*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_108
˘
IsVariableInitialized_160IsVariableInitializedtraining/Adam/Variable_109*-
_class#
!loc:@training/Adam/Variable_109*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_161IsVariableInitializedtraining/Adam/Variable_110*-
_class#
!loc:@training/Adam/Variable_110*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_162IsVariableInitializedtraining/Adam/Variable_111*-
_class#
!loc:@training/Adam/Variable_111*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_163IsVariableInitializedtraining/Adam/Variable_112*
dtype0*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_112
˘
IsVariableInitialized_164IsVariableInitializedtraining/Adam/Variable_113*-
_class#
!loc:@training/Adam/Variable_113*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_165IsVariableInitializedtraining/Adam/Variable_114*
dtype0*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_114
˘
IsVariableInitialized_166IsVariableInitializedtraining/Adam/Variable_115*
dtype0*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_115
˘
IsVariableInitialized_167IsVariableInitializedtraining/Adam/Variable_116*-
_class#
!loc:@training/Adam/Variable_116*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_168IsVariableInitializedtraining/Adam/Variable_117*-
_class#
!loc:@training/Adam/Variable_117*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_169IsVariableInitializedtraining/Adam/Variable_118*-
_class#
!loc:@training/Adam/Variable_118*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_170IsVariableInitializedtraining/Adam/Variable_119*-
_class#
!loc:@training/Adam/Variable_119*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_171IsVariableInitializedtraining/Adam/Variable_120*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_120*
dtype0
˘
IsVariableInitialized_172IsVariableInitializedtraining/Adam/Variable_121*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_121*
dtype0
˘
IsVariableInitialized_173IsVariableInitializedtraining/Adam/Variable_122*-
_class#
!loc:@training/Adam/Variable_122*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_174IsVariableInitializedtraining/Adam/Variable_123*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_123*
dtype0
˘
IsVariableInitialized_175IsVariableInitializedtraining/Adam/Variable_124*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_124*
dtype0
˘
IsVariableInitialized_176IsVariableInitializedtraining/Adam/Variable_125*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_125*
dtype0
˘
IsVariableInitialized_177IsVariableInitializedtraining/Adam/Variable_126*
dtype0*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_126
˘
IsVariableInitialized_178IsVariableInitializedtraining/Adam/Variable_127*-
_class#
!loc:@training/Adam/Variable_127*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_179IsVariableInitializedtraining/Adam/Variable_128*
dtype0*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_128
˘
IsVariableInitialized_180IsVariableInitializedtraining/Adam/Variable_129*-
_class#
!loc:@training/Adam/Variable_129*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_181IsVariableInitializedtraining/Adam/Variable_130*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_130*
dtype0
˘
IsVariableInitialized_182IsVariableInitializedtraining/Adam/Variable_131*-
_class#
!loc:@training/Adam/Variable_131*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_183IsVariableInitializedtraining/Adam/Variable_132*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_132*
dtype0
˘
IsVariableInitialized_184IsVariableInitializedtraining/Adam/Variable_133*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_133*
dtype0
˘
IsVariableInitialized_185IsVariableInitializedtraining/Adam/Variable_134*-
_class#
!loc:@training/Adam/Variable_134*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_186IsVariableInitializedtraining/Adam/Variable_135*-
_class#
!loc:@training/Adam/Variable_135*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_187IsVariableInitializedtraining/Adam/Variable_136*-
_class#
!loc:@training/Adam/Variable_136*
dtype0*
_output_shapes
: 
˘
IsVariableInitialized_188IsVariableInitializedtraining/Adam/Variable_137*
_output_shapes
: *-
_class#
!loc:@training/Adam/Variable_137*
dtype0
/
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^conv1a/bias/Assign^conv1a/kernel/Assign^conv1b/bias/Assign^conv1b/kernel/Assign^conv2a/bias/Assign^conv2a/kernel/Assign^conv2b/bias/Assign^conv2b/kernel/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_10/bias/Assign^conv2d_10/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^conv2d_3/bias/Assign^conv2d_3/kernel/Assign^conv2d_4/bias/Assign^conv2d_4/kernel/Assign^conv2d_5/bias/Assign^conv2d_5/kernel/Assign^conv2d_6/bias/Assign^conv2d_6/kernel/Assign^conv2d_7/bias/Assign^conv2d_7/kernel/Assign^conv2d_8/bias/Assign^conv2d_8/kernel/Assign^conv2d_9/bias/Assign^conv2d_9/kernel/Assign^conv3a/bias/Assign^conv3a/kernel/Assign^conv3b/bias/Assign^conv3b/kernel/Assign^conv4a/bias/Assign^conv4a/kernel/Assign^conv4b/bias/Assign^conv4b/kernel/Assign^conv5a/bias/Assign^conv5a/kernel/Assign^conv5b/bias/Assign^conv5b/kernel/Assign^conv6/bias/Assign^conv6/kernel/Assign^conv7a/bias/Assign^conv7a/kernel/Assign^conv7b/bias/Assign^conv7b/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign"^training/Adam/Variable_100/Assign"^training/Adam/Variable_101/Assign"^training/Adam/Variable_102/Assign"^training/Adam/Variable_103/Assign"^training/Adam/Variable_104/Assign"^training/Adam/Variable_105/Assign"^training/Adam/Variable_106/Assign"^training/Adam/Variable_107/Assign"^training/Adam/Variable_108/Assign"^training/Adam/Variable_109/Assign!^training/Adam/Variable_11/Assign"^training/Adam/Variable_110/Assign"^training/Adam/Variable_111/Assign"^training/Adam/Variable_112/Assign"^training/Adam/Variable_113/Assign"^training/Adam/Variable_114/Assign"^training/Adam/Variable_115/Assign"^training/Adam/Variable_116/Assign"^training/Adam/Variable_117/Assign"^training/Adam/Variable_118/Assign"^training/Adam/Variable_119/Assign!^training/Adam/Variable_12/Assign"^training/Adam/Variable_120/Assign"^training/Adam/Variable_121/Assign"^training/Adam/Variable_122/Assign"^training/Adam/Variable_123/Assign"^training/Adam/Variable_124/Assign"^training/Adam/Variable_125/Assign"^training/Adam/Variable_126/Assign"^training/Adam/Variable_127/Assign"^training/Adam/Variable_128/Assign"^training/Adam/Variable_129/Assign!^training/Adam/Variable_13/Assign"^training/Adam/Variable_130/Assign"^training/Adam/Variable_131/Assign"^training/Adam/Variable_132/Assign"^training/Adam/Variable_133/Assign"^training/Adam/Variable_134/Assign"^training/Adam/Variable_135/Assign"^training/Adam/Variable_136/Assign"^training/Adam/Variable_137/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign!^training/Adam/Variable_24/Assign!^training/Adam/Variable_25/Assign!^training/Adam/Variable_26/Assign!^training/Adam/Variable_27/Assign!^training/Adam/Variable_28/Assign!^training/Adam/Variable_29/Assign ^training/Adam/Variable_3/Assign!^training/Adam/Variable_30/Assign!^training/Adam/Variable_31/Assign!^training/Adam/Variable_32/Assign!^training/Adam/Variable_33/Assign!^training/Adam/Variable_34/Assign!^training/Adam/Variable_35/Assign!^training/Adam/Variable_36/Assign!^training/Adam/Variable_37/Assign!^training/Adam/Variable_38/Assign!^training/Adam/Variable_39/Assign ^training/Adam/Variable_4/Assign!^training/Adam/Variable_40/Assign!^training/Adam/Variable_41/Assign!^training/Adam/Variable_42/Assign!^training/Adam/Variable_43/Assign!^training/Adam/Variable_44/Assign!^training/Adam/Variable_45/Assign!^training/Adam/Variable_46/Assign!^training/Adam/Variable_47/Assign!^training/Adam/Variable_48/Assign!^training/Adam/Variable_49/Assign ^training/Adam/Variable_5/Assign!^training/Adam/Variable_50/Assign!^training/Adam/Variable_51/Assign!^training/Adam/Variable_52/Assign!^training/Adam/Variable_53/Assign!^training/Adam/Variable_54/Assign!^training/Adam/Variable_55/Assign!^training/Adam/Variable_56/Assign!^training/Adam/Variable_57/Assign!^training/Adam/Variable_58/Assign!^training/Adam/Variable_59/Assign ^training/Adam/Variable_6/Assign!^training/Adam/Variable_60/Assign!^training/Adam/Variable_61/Assign!^training/Adam/Variable_62/Assign!^training/Adam/Variable_63/Assign!^training/Adam/Variable_64/Assign!^training/Adam/Variable_65/Assign!^training/Adam/Variable_66/Assign!^training/Adam/Variable_67/Assign!^training/Adam/Variable_68/Assign!^training/Adam/Variable_69/Assign ^training/Adam/Variable_7/Assign!^training/Adam/Variable_70/Assign!^training/Adam/Variable_71/Assign!^training/Adam/Variable_72/Assign!^training/Adam/Variable_73/Assign!^training/Adam/Variable_74/Assign!^training/Adam/Variable_75/Assign!^training/Adam/Variable_76/Assign!^training/Adam/Variable_77/Assign!^training/Adam/Variable_78/Assign!^training/Adam/Variable_79/Assign ^training/Adam/Variable_8/Assign!^training/Adam/Variable_80/Assign!^training/Adam/Variable_81/Assign!^training/Adam/Variable_82/Assign!^training/Adam/Variable_83/Assign!^training/Adam/Variable_84/Assign!^training/Adam/Variable_85/Assign!^training/Adam/Variable_86/Assign!^training/Adam/Variable_87/Assign!^training/Adam/Variable_88/Assign!^training/Adam/Variable_89/Assign ^training/Adam/Variable_9/Assign!^training/Adam/Variable_90/Assign!^training/Adam/Variable_91/Assign!^training/Adam/Variable_92/Assign!^training/Adam/Variable_93/Assign!^training/Adam/Variable_94/Assign!^training/Adam/Variable_95/Assign!^training/Adam/Variable_96/Assign!^training/Adam/Variable_97/Assign!^training/Adam/Variable_98/Assign!^training/Adam/Variable_99/Assign""ôŹ
trainable_variablesŰŹ×Ź
Z
conv1a/kernel:0conv1a/kernel/Assignconv1a/kernel/read:02conv1a/truncated_normal:08
I
conv1a/bias:0conv1a/bias/Assignconv1a/bias/read:02conv1a/Const:08
Z
conv1b/kernel:0conv1b/kernel/Assignconv1b/kernel/read:02conv1b/truncated_normal:08
I
conv1b/bias:0conv1b/bias/Assignconv1b/bias/read:02conv1b/Const:08
Z
conv2a/kernel:0conv2a/kernel/Assignconv2a/kernel/read:02conv2a/truncated_normal:08
I
conv2a/bias:0conv2a/bias/Assignconv2a/bias/read:02conv2a/Const:08
Z
conv2b/kernel:0conv2b/kernel/Assignconv2b/kernel/read:02conv2b/truncated_normal:08
I
conv2b/bias:0conv2b/bias/Assignconv2b/bias/read:02conv2b/Const:08
Z
conv3a/kernel:0conv3a/kernel/Assignconv3a/kernel/read:02conv3a/truncated_normal:08
I
conv3a/bias:0conv3a/bias/Assignconv3a/bias/read:02conv3a/Const:08
Z
conv3b/kernel:0conv3b/kernel/Assignconv3b/kernel/read:02conv3b/truncated_normal:08
I
conv3b/bias:0conv3b/bias/Assignconv3b/bias/read:02conv3b/Const:08
Z
conv4a/kernel:0conv4a/kernel/Assignconv4a/kernel/read:02conv4a/truncated_normal:08
I
conv4a/bias:0conv4a/bias/Assignconv4a/bias/read:02conv4a/Const:08
Z
conv4b/kernel:0conv4b/kernel/Assignconv4b/kernel/read:02conv4b/truncated_normal:08
I
conv4b/bias:0conv4b/bias/Assignconv4b/bias/read:02conv4b/Const:08
Z
conv5a/kernel:0conv5a/kernel/Assignconv5a/kernel/read:02conv5a/truncated_normal:08
I
conv5a/bias:0conv5a/bias/Assignconv5a/bias/read:02conv5a/Const:08
Z
conv5b/kernel:0conv5b/kernel/Assignconv5b/kernel/read:02conv5b/truncated_normal:08
I
conv5b/bias:0conv5b/bias/Assignconv5b/bias/read:02conv5b/Const:08
V
conv6/kernel:0conv6/kernel/Assignconv6/kernel/read:02conv6/truncated_normal:08
E
conv6/bias:0conv6/bias/Assignconv6/bias/read:02conv6/Const:08
Z
conv7a/kernel:0conv7a/kernel/Assignconv7a/kernel/read:02conv7a/truncated_normal:08
I
conv7a/bias:0conv7a/bias/Assignconv7a/bias/read:02conv7a/Const:08
Z
conv7b/kernel:0conv7b/kernel/Assignconv7b/kernel/read:02conv7b/truncated_normal:08
I
conv7b/bias:0conv7b/bias/Assignconv7b/bias/read:02conv7b/Const:08
b
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02conv2d_1/truncated_normal:08
Q
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02conv2d_1/Const:08
b
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02conv2d_2/truncated_normal:08
Q
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:02conv2d_2/Const:08
b
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:02conv2d_3/truncated_normal:08
Q
conv2d_3/bias:0conv2d_3/bias/Assignconv2d_3/bias/read:02conv2d_3/Const:08
b
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:02conv2d_4/truncated_normal:08
Q
conv2d_4/bias:0conv2d_4/bias/Assignconv2d_4/bias/read:02conv2d_4/Const:08
b
conv2d_5/kernel:0conv2d_5/kernel/Assignconv2d_5/kernel/read:02conv2d_5/truncated_normal:08
Q
conv2d_5/bias:0conv2d_5/bias/Assignconv2d_5/bias/read:02conv2d_5/Const:08
b
conv2d_6/kernel:0conv2d_6/kernel/Assignconv2d_6/kernel/read:02conv2d_6/truncated_normal:08
Q
conv2d_6/bias:0conv2d_6/bias/Assignconv2d_6/bias/read:02conv2d_6/Const:08
b
conv2d_7/kernel:0conv2d_7/kernel/Assignconv2d_7/kernel/read:02conv2d_7/truncated_normal:08
Q
conv2d_7/bias:0conv2d_7/bias/Assignconv2d_7/bias/read:02conv2d_7/Const:08
b
conv2d_8/kernel:0conv2d_8/kernel/Assignconv2d_8/kernel/read:02conv2d_8/truncated_normal:08
Q
conv2d_8/bias:0conv2d_8/bias/Assignconv2d_8/bias/read:02conv2d_8/Const:08
b
conv2d_9/kernel:0conv2d_9/kernel/Assignconv2d_9/kernel/read:02conv2d_9/truncated_normal:08
Q
conv2d_9/bias:0conv2d_9/bias/Assignconv2d_9/bias/read:02conv2d_9/Const:08
d
conv2d_10/kernel:0conv2d_10/kernel/Assignconv2d_10/kernel/read:02conv2d_10/random_uniform:08
U
conv2d_10/bias:0conv2d_10/bias/Assignconv2d_10/bias/read:02conv2d_10/Const:08
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
training/Adam/Variable_41:0 training/Adam/Variable_41/Assign training/Adam/Variable_41/read:02training/Adam/zeros_41:08
}
training/Adam/Variable_42:0 training/Adam/Variable_42/Assign training/Adam/Variable_42/read:02training/Adam/zeros_42:08
}
training/Adam/Variable_43:0 training/Adam/Variable_43/Assign training/Adam/Variable_43/read:02training/Adam/zeros_43:08
}
training/Adam/Variable_44:0 training/Adam/Variable_44/Assign training/Adam/Variable_44/read:02training/Adam/zeros_44:08
}
training/Adam/Variable_45:0 training/Adam/Variable_45/Assign training/Adam/Variable_45/read:02training/Adam/zeros_45:08
}
training/Adam/Variable_46:0 training/Adam/Variable_46/Assign training/Adam/Variable_46/read:02training/Adam/zeros_46:08
}
training/Adam/Variable_47:0 training/Adam/Variable_47/Assign training/Adam/Variable_47/read:02training/Adam/zeros_47:08
}
training/Adam/Variable_48:0 training/Adam/Variable_48/Assign training/Adam/Variable_48/read:02training/Adam/zeros_48:08
}
training/Adam/Variable_49:0 training/Adam/Variable_49/Assign training/Adam/Variable_49/read:02training/Adam/zeros_49:08
}
training/Adam/Variable_50:0 training/Adam/Variable_50/Assign training/Adam/Variable_50/read:02training/Adam/zeros_50:08
}
training/Adam/Variable_51:0 training/Adam/Variable_51/Assign training/Adam/Variable_51/read:02training/Adam/zeros_51:08
}
training/Adam/Variable_52:0 training/Adam/Variable_52/Assign training/Adam/Variable_52/read:02training/Adam/zeros_52:08
}
training/Adam/Variable_53:0 training/Adam/Variable_53/Assign training/Adam/Variable_53/read:02training/Adam/zeros_53:08
}
training/Adam/Variable_54:0 training/Adam/Variable_54/Assign training/Adam/Variable_54/read:02training/Adam/zeros_54:08
}
training/Adam/Variable_55:0 training/Adam/Variable_55/Assign training/Adam/Variable_55/read:02training/Adam/zeros_55:08
}
training/Adam/Variable_56:0 training/Adam/Variable_56/Assign training/Adam/Variable_56/read:02training/Adam/zeros_56:08
}
training/Adam/Variable_57:0 training/Adam/Variable_57/Assign training/Adam/Variable_57/read:02training/Adam/zeros_57:08
}
training/Adam/Variable_58:0 training/Adam/Variable_58/Assign training/Adam/Variable_58/read:02training/Adam/zeros_58:08
}
training/Adam/Variable_59:0 training/Adam/Variable_59/Assign training/Adam/Variable_59/read:02training/Adam/zeros_59:08
}
training/Adam/Variable_60:0 training/Adam/Variable_60/Assign training/Adam/Variable_60/read:02training/Adam/zeros_60:08
}
training/Adam/Variable_61:0 training/Adam/Variable_61/Assign training/Adam/Variable_61/read:02training/Adam/zeros_61:08
}
training/Adam/Variable_62:0 training/Adam/Variable_62/Assign training/Adam/Variable_62/read:02training/Adam/zeros_62:08
}
training/Adam/Variable_63:0 training/Adam/Variable_63/Assign training/Adam/Variable_63/read:02training/Adam/zeros_63:08
}
training/Adam/Variable_64:0 training/Adam/Variable_64/Assign training/Adam/Variable_64/read:02training/Adam/zeros_64:08
}
training/Adam/Variable_65:0 training/Adam/Variable_65/Assign training/Adam/Variable_65/read:02training/Adam/zeros_65:08
}
training/Adam/Variable_66:0 training/Adam/Variable_66/Assign training/Adam/Variable_66/read:02training/Adam/zeros_66:08
}
training/Adam/Variable_67:0 training/Adam/Variable_67/Assign training/Adam/Variable_67/read:02training/Adam/zeros_67:08
}
training/Adam/Variable_68:0 training/Adam/Variable_68/Assign training/Adam/Variable_68/read:02training/Adam/zeros_68:08
}
training/Adam/Variable_69:0 training/Adam/Variable_69/Assign training/Adam/Variable_69/read:02training/Adam/zeros_69:08
}
training/Adam/Variable_70:0 training/Adam/Variable_70/Assign training/Adam/Variable_70/read:02training/Adam/zeros_70:08
}
training/Adam/Variable_71:0 training/Adam/Variable_71/Assign training/Adam/Variable_71/read:02training/Adam/zeros_71:08
}
training/Adam/Variable_72:0 training/Adam/Variable_72/Assign training/Adam/Variable_72/read:02training/Adam/zeros_72:08
}
training/Adam/Variable_73:0 training/Adam/Variable_73/Assign training/Adam/Variable_73/read:02training/Adam/zeros_73:08
}
training/Adam/Variable_74:0 training/Adam/Variable_74/Assign training/Adam/Variable_74/read:02training/Adam/zeros_74:08
}
training/Adam/Variable_75:0 training/Adam/Variable_75/Assign training/Adam/Variable_75/read:02training/Adam/zeros_75:08
}
training/Adam/Variable_76:0 training/Adam/Variable_76/Assign training/Adam/Variable_76/read:02training/Adam/zeros_76:08
}
training/Adam/Variable_77:0 training/Adam/Variable_77/Assign training/Adam/Variable_77/read:02training/Adam/zeros_77:08
}
training/Adam/Variable_78:0 training/Adam/Variable_78/Assign training/Adam/Variable_78/read:02training/Adam/zeros_78:08
}
training/Adam/Variable_79:0 training/Adam/Variable_79/Assign training/Adam/Variable_79/read:02training/Adam/zeros_79:08
}
training/Adam/Variable_80:0 training/Adam/Variable_80/Assign training/Adam/Variable_80/read:02training/Adam/zeros_80:08
}
training/Adam/Variable_81:0 training/Adam/Variable_81/Assign training/Adam/Variable_81/read:02training/Adam/zeros_81:08
}
training/Adam/Variable_82:0 training/Adam/Variable_82/Assign training/Adam/Variable_82/read:02training/Adam/zeros_82:08
}
training/Adam/Variable_83:0 training/Adam/Variable_83/Assign training/Adam/Variable_83/read:02training/Adam/zeros_83:08
}
training/Adam/Variable_84:0 training/Adam/Variable_84/Assign training/Adam/Variable_84/read:02training/Adam/zeros_84:08
}
training/Adam/Variable_85:0 training/Adam/Variable_85/Assign training/Adam/Variable_85/read:02training/Adam/zeros_85:08
}
training/Adam/Variable_86:0 training/Adam/Variable_86/Assign training/Adam/Variable_86/read:02training/Adam/zeros_86:08
}
training/Adam/Variable_87:0 training/Adam/Variable_87/Assign training/Adam/Variable_87/read:02training/Adam/zeros_87:08
}
training/Adam/Variable_88:0 training/Adam/Variable_88/Assign training/Adam/Variable_88/read:02training/Adam/zeros_88:08
}
training/Adam/Variable_89:0 training/Adam/Variable_89/Assign training/Adam/Variable_89/read:02training/Adam/zeros_89:08
}
training/Adam/Variable_90:0 training/Adam/Variable_90/Assign training/Adam/Variable_90/read:02training/Adam/zeros_90:08
}
training/Adam/Variable_91:0 training/Adam/Variable_91/Assign training/Adam/Variable_91/read:02training/Adam/zeros_91:08
}
training/Adam/Variable_92:0 training/Adam/Variable_92/Assign training/Adam/Variable_92/read:02training/Adam/zeros_92:08
}
training/Adam/Variable_93:0 training/Adam/Variable_93/Assign training/Adam/Variable_93/read:02training/Adam/zeros_93:08
}
training/Adam/Variable_94:0 training/Adam/Variable_94/Assign training/Adam/Variable_94/read:02training/Adam/zeros_94:08
}
training/Adam/Variable_95:0 training/Adam/Variable_95/Assign training/Adam/Variable_95/read:02training/Adam/zeros_95:08
}
training/Adam/Variable_96:0 training/Adam/Variable_96/Assign training/Adam/Variable_96/read:02training/Adam/zeros_96:08
}
training/Adam/Variable_97:0 training/Adam/Variable_97/Assign training/Adam/Variable_97/read:02training/Adam/zeros_97:08
}
training/Adam/Variable_98:0 training/Adam/Variable_98/Assign training/Adam/Variable_98/read:02training/Adam/zeros_98:08
}
training/Adam/Variable_99:0 training/Adam/Variable_99/Assign training/Adam/Variable_99/read:02training/Adam/zeros_99:08

training/Adam/Variable_100:0!training/Adam/Variable_100/Assign!training/Adam/Variable_100/read:02training/Adam/zeros_100:08

training/Adam/Variable_101:0!training/Adam/Variable_101/Assign!training/Adam/Variable_101/read:02training/Adam/zeros_101:08

training/Adam/Variable_102:0!training/Adam/Variable_102/Assign!training/Adam/Variable_102/read:02training/Adam/zeros_102:08

training/Adam/Variable_103:0!training/Adam/Variable_103/Assign!training/Adam/Variable_103/read:02training/Adam/zeros_103:08

training/Adam/Variable_104:0!training/Adam/Variable_104/Assign!training/Adam/Variable_104/read:02training/Adam/zeros_104:08

training/Adam/Variable_105:0!training/Adam/Variable_105/Assign!training/Adam/Variable_105/read:02training/Adam/zeros_105:08

training/Adam/Variable_106:0!training/Adam/Variable_106/Assign!training/Adam/Variable_106/read:02training/Adam/zeros_106:08

training/Adam/Variable_107:0!training/Adam/Variable_107/Assign!training/Adam/Variable_107/read:02training/Adam/zeros_107:08

training/Adam/Variable_108:0!training/Adam/Variable_108/Assign!training/Adam/Variable_108/read:02training/Adam/zeros_108:08

training/Adam/Variable_109:0!training/Adam/Variable_109/Assign!training/Adam/Variable_109/read:02training/Adam/zeros_109:08

training/Adam/Variable_110:0!training/Adam/Variable_110/Assign!training/Adam/Variable_110/read:02training/Adam/zeros_110:08

training/Adam/Variable_111:0!training/Adam/Variable_111/Assign!training/Adam/Variable_111/read:02training/Adam/zeros_111:08

training/Adam/Variable_112:0!training/Adam/Variable_112/Assign!training/Adam/Variable_112/read:02training/Adam/zeros_112:08

training/Adam/Variable_113:0!training/Adam/Variable_113/Assign!training/Adam/Variable_113/read:02training/Adam/zeros_113:08

training/Adam/Variable_114:0!training/Adam/Variable_114/Assign!training/Adam/Variable_114/read:02training/Adam/zeros_114:08

training/Adam/Variable_115:0!training/Adam/Variable_115/Assign!training/Adam/Variable_115/read:02training/Adam/zeros_115:08

training/Adam/Variable_116:0!training/Adam/Variable_116/Assign!training/Adam/Variable_116/read:02training/Adam/zeros_116:08

training/Adam/Variable_117:0!training/Adam/Variable_117/Assign!training/Adam/Variable_117/read:02training/Adam/zeros_117:08

training/Adam/Variable_118:0!training/Adam/Variable_118/Assign!training/Adam/Variable_118/read:02training/Adam/zeros_118:08

training/Adam/Variable_119:0!training/Adam/Variable_119/Assign!training/Adam/Variable_119/read:02training/Adam/zeros_119:08

training/Adam/Variable_120:0!training/Adam/Variable_120/Assign!training/Adam/Variable_120/read:02training/Adam/zeros_120:08

training/Adam/Variable_121:0!training/Adam/Variable_121/Assign!training/Adam/Variable_121/read:02training/Adam/zeros_121:08

training/Adam/Variable_122:0!training/Adam/Variable_122/Assign!training/Adam/Variable_122/read:02training/Adam/zeros_122:08

training/Adam/Variable_123:0!training/Adam/Variable_123/Assign!training/Adam/Variable_123/read:02training/Adam/zeros_123:08

training/Adam/Variable_124:0!training/Adam/Variable_124/Assign!training/Adam/Variable_124/read:02training/Adam/zeros_124:08

training/Adam/Variable_125:0!training/Adam/Variable_125/Assign!training/Adam/Variable_125/read:02training/Adam/zeros_125:08

training/Adam/Variable_126:0!training/Adam/Variable_126/Assign!training/Adam/Variable_126/read:02training/Adam/zeros_126:08

training/Adam/Variable_127:0!training/Adam/Variable_127/Assign!training/Adam/Variable_127/read:02training/Adam/zeros_127:08

training/Adam/Variable_128:0!training/Adam/Variable_128/Assign!training/Adam/Variable_128/read:02training/Adam/zeros_128:08

training/Adam/Variable_129:0!training/Adam/Variable_129/Assign!training/Adam/Variable_129/read:02training/Adam/zeros_129:08

training/Adam/Variable_130:0!training/Adam/Variable_130/Assign!training/Adam/Variable_130/read:02training/Adam/zeros_130:08

training/Adam/Variable_131:0!training/Adam/Variable_131/Assign!training/Adam/Variable_131/read:02training/Adam/zeros_131:08

training/Adam/Variable_132:0!training/Adam/Variable_132/Assign!training/Adam/Variable_132/read:02training/Adam/zeros_132:08

training/Adam/Variable_133:0!training/Adam/Variable_133/Assign!training/Adam/Variable_133/read:02training/Adam/zeros_133:08

training/Adam/Variable_134:0!training/Adam/Variable_134/Assign!training/Adam/Variable_134/read:02training/Adam/zeros_134:08

training/Adam/Variable_135:0!training/Adam/Variable_135/Assign!training/Adam/Variable_135/read:02training/Adam/zeros_135:08

training/Adam/Variable_136:0!training/Adam/Variable_136/Assign!training/Adam/Variable_136/read:02training/Adam/zeros_136:08

training/Adam/Variable_137:0!training/Adam/Variable_137/Assign!training/Adam/Variable_137/read:02training/Adam/zeros_137:08"
cond_context
ň
drop4/cond/cond_textdrop4/cond/pred_id:0drop4/cond/switch_t:0 *Ş
conv4b/Relu:0
drop4/cond/dropout/Floor:0
drop4/cond/dropout/Shape:0
drop4/cond/dropout/add:0
drop4/cond/dropout/mul:0
1drop4/cond/dropout/random_uniform/RandomUniform:0
'drop4/cond/dropout/random_uniform/max:0
'drop4/cond/dropout/random_uniform/min:0
'drop4/cond/dropout/random_uniform/mul:0
'drop4/cond/dropout/random_uniform/sub:0
#drop4/cond/dropout/random_uniform:0
drop4/cond/dropout/rate:0
drop4/cond/dropout/sub/x:0
drop4/cond/dropout/sub:0
drop4/cond/dropout/truediv:0
drop4/cond/mul/Switch:1
drop4/cond/mul/y:0
drop4/cond/mul:0
drop4/cond/pred_id:0
drop4/cond/switch_t:0,
drop4/cond/pred_id:0drop4/cond/pred_id:0(
conv4b/Relu:0drop4/cond/mul/Switch:1

drop4/cond/cond_text_1drop4/cond/pred_id:0drop4/cond/switch_f:0*Ŕ
conv4b/Relu:0
drop4/cond/Switch_1:0
drop4/cond/Switch_1:1
drop4/cond/pred_id:0
drop4/cond/switch_f:0&
conv4b/Relu:0drop4/cond/Switch_1:0,
drop4/cond/pred_id:0drop4/cond/pred_id:0
ň
drop5/cond/cond_textdrop5/cond/pred_id:0drop5/cond/switch_t:0 *Ş
conv5b/Relu:0
drop5/cond/dropout/Floor:0
drop5/cond/dropout/Shape:0
drop5/cond/dropout/add:0
drop5/cond/dropout/mul:0
1drop5/cond/dropout/random_uniform/RandomUniform:0
'drop5/cond/dropout/random_uniform/max:0
'drop5/cond/dropout/random_uniform/min:0
'drop5/cond/dropout/random_uniform/mul:0
'drop5/cond/dropout/random_uniform/sub:0
#drop5/cond/dropout/random_uniform:0
drop5/cond/dropout/rate:0
drop5/cond/dropout/sub/x:0
drop5/cond/dropout/sub:0
drop5/cond/dropout/truediv:0
drop5/cond/mul/Switch:1
drop5/cond/mul/y:0
drop5/cond/mul:0
drop5/cond/pred_id:0
drop5/cond/switch_t:0,
drop5/cond/pred_id:0drop5/cond/pred_id:0(
conv5b/Relu:0drop5/cond/mul/Switch:1

drop5/cond/cond_text_1drop5/cond/pred_id:0drop5/cond/switch_f:0*Ŕ
conv5b/Relu:0
drop5/cond/Switch_1:0
drop5/cond/Switch_1:1
drop5/cond/pred_id:0
drop5/cond/switch_f:0,
drop5/cond/pred_id:0drop5/cond/pred_id:0&
conv5b/Relu:0drop5/cond/Switch_1:0"ęŹ
	variablesŰŹ×Ź
Z
conv1a/kernel:0conv1a/kernel/Assignconv1a/kernel/read:02conv1a/truncated_normal:08
I
conv1a/bias:0conv1a/bias/Assignconv1a/bias/read:02conv1a/Const:08
Z
conv1b/kernel:0conv1b/kernel/Assignconv1b/kernel/read:02conv1b/truncated_normal:08
I
conv1b/bias:0conv1b/bias/Assignconv1b/bias/read:02conv1b/Const:08
Z
conv2a/kernel:0conv2a/kernel/Assignconv2a/kernel/read:02conv2a/truncated_normal:08
I
conv2a/bias:0conv2a/bias/Assignconv2a/bias/read:02conv2a/Const:08
Z
conv2b/kernel:0conv2b/kernel/Assignconv2b/kernel/read:02conv2b/truncated_normal:08
I
conv2b/bias:0conv2b/bias/Assignconv2b/bias/read:02conv2b/Const:08
Z
conv3a/kernel:0conv3a/kernel/Assignconv3a/kernel/read:02conv3a/truncated_normal:08
I
conv3a/bias:0conv3a/bias/Assignconv3a/bias/read:02conv3a/Const:08
Z
conv3b/kernel:0conv3b/kernel/Assignconv3b/kernel/read:02conv3b/truncated_normal:08
I
conv3b/bias:0conv3b/bias/Assignconv3b/bias/read:02conv3b/Const:08
Z
conv4a/kernel:0conv4a/kernel/Assignconv4a/kernel/read:02conv4a/truncated_normal:08
I
conv4a/bias:0conv4a/bias/Assignconv4a/bias/read:02conv4a/Const:08
Z
conv4b/kernel:0conv4b/kernel/Assignconv4b/kernel/read:02conv4b/truncated_normal:08
I
conv4b/bias:0conv4b/bias/Assignconv4b/bias/read:02conv4b/Const:08
Z
conv5a/kernel:0conv5a/kernel/Assignconv5a/kernel/read:02conv5a/truncated_normal:08
I
conv5a/bias:0conv5a/bias/Assignconv5a/bias/read:02conv5a/Const:08
Z
conv5b/kernel:0conv5b/kernel/Assignconv5b/kernel/read:02conv5b/truncated_normal:08
I
conv5b/bias:0conv5b/bias/Assignconv5b/bias/read:02conv5b/Const:08
V
conv6/kernel:0conv6/kernel/Assignconv6/kernel/read:02conv6/truncated_normal:08
E
conv6/bias:0conv6/bias/Assignconv6/bias/read:02conv6/Const:08
Z
conv7a/kernel:0conv7a/kernel/Assignconv7a/kernel/read:02conv7a/truncated_normal:08
I
conv7a/bias:0conv7a/bias/Assignconv7a/bias/read:02conv7a/Const:08
Z
conv7b/kernel:0conv7b/kernel/Assignconv7b/kernel/read:02conv7b/truncated_normal:08
I
conv7b/bias:0conv7b/bias/Assignconv7b/bias/read:02conv7b/Const:08
b
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02conv2d_1/truncated_normal:08
Q
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02conv2d_1/Const:08
b
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02conv2d_2/truncated_normal:08
Q
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:02conv2d_2/Const:08
b
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:02conv2d_3/truncated_normal:08
Q
conv2d_3/bias:0conv2d_3/bias/Assignconv2d_3/bias/read:02conv2d_3/Const:08
b
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:02conv2d_4/truncated_normal:08
Q
conv2d_4/bias:0conv2d_4/bias/Assignconv2d_4/bias/read:02conv2d_4/Const:08
b
conv2d_5/kernel:0conv2d_5/kernel/Assignconv2d_5/kernel/read:02conv2d_5/truncated_normal:08
Q
conv2d_5/bias:0conv2d_5/bias/Assignconv2d_5/bias/read:02conv2d_5/Const:08
b
conv2d_6/kernel:0conv2d_6/kernel/Assignconv2d_6/kernel/read:02conv2d_6/truncated_normal:08
Q
conv2d_6/bias:0conv2d_6/bias/Assignconv2d_6/bias/read:02conv2d_6/Const:08
b
conv2d_7/kernel:0conv2d_7/kernel/Assignconv2d_7/kernel/read:02conv2d_7/truncated_normal:08
Q
conv2d_7/bias:0conv2d_7/bias/Assignconv2d_7/bias/read:02conv2d_7/Const:08
b
conv2d_8/kernel:0conv2d_8/kernel/Assignconv2d_8/kernel/read:02conv2d_8/truncated_normal:08
Q
conv2d_8/bias:0conv2d_8/bias/Assignconv2d_8/bias/read:02conv2d_8/Const:08
b
conv2d_9/kernel:0conv2d_9/kernel/Assignconv2d_9/kernel/read:02conv2d_9/truncated_normal:08
Q
conv2d_9/bias:0conv2d_9/bias/Assignconv2d_9/bias/read:02conv2d_9/Const:08
d
conv2d_10/kernel:0conv2d_10/kernel/Assignconv2d_10/kernel/read:02conv2d_10/random_uniform:08
U
conv2d_10/bias:0conv2d_10/bias/Assignconv2d_10/bias/read:02conv2d_10/Const:08
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
training/Adam/Variable_41:0 training/Adam/Variable_41/Assign training/Adam/Variable_41/read:02training/Adam/zeros_41:08
}
training/Adam/Variable_42:0 training/Adam/Variable_42/Assign training/Adam/Variable_42/read:02training/Adam/zeros_42:08
}
training/Adam/Variable_43:0 training/Adam/Variable_43/Assign training/Adam/Variable_43/read:02training/Adam/zeros_43:08
}
training/Adam/Variable_44:0 training/Adam/Variable_44/Assign training/Adam/Variable_44/read:02training/Adam/zeros_44:08
}
training/Adam/Variable_45:0 training/Adam/Variable_45/Assign training/Adam/Variable_45/read:02training/Adam/zeros_45:08
}
training/Adam/Variable_46:0 training/Adam/Variable_46/Assign training/Adam/Variable_46/read:02training/Adam/zeros_46:08
}
training/Adam/Variable_47:0 training/Adam/Variable_47/Assign training/Adam/Variable_47/read:02training/Adam/zeros_47:08
}
training/Adam/Variable_48:0 training/Adam/Variable_48/Assign training/Adam/Variable_48/read:02training/Adam/zeros_48:08
}
training/Adam/Variable_49:0 training/Adam/Variable_49/Assign training/Adam/Variable_49/read:02training/Adam/zeros_49:08
}
training/Adam/Variable_50:0 training/Adam/Variable_50/Assign training/Adam/Variable_50/read:02training/Adam/zeros_50:08
}
training/Adam/Variable_51:0 training/Adam/Variable_51/Assign training/Adam/Variable_51/read:02training/Adam/zeros_51:08
}
training/Adam/Variable_52:0 training/Adam/Variable_52/Assign training/Adam/Variable_52/read:02training/Adam/zeros_52:08
}
training/Adam/Variable_53:0 training/Adam/Variable_53/Assign training/Adam/Variable_53/read:02training/Adam/zeros_53:08
}
training/Adam/Variable_54:0 training/Adam/Variable_54/Assign training/Adam/Variable_54/read:02training/Adam/zeros_54:08
}
training/Adam/Variable_55:0 training/Adam/Variable_55/Assign training/Adam/Variable_55/read:02training/Adam/zeros_55:08
}
training/Adam/Variable_56:0 training/Adam/Variable_56/Assign training/Adam/Variable_56/read:02training/Adam/zeros_56:08
}
training/Adam/Variable_57:0 training/Adam/Variable_57/Assign training/Adam/Variable_57/read:02training/Adam/zeros_57:08
}
training/Adam/Variable_58:0 training/Adam/Variable_58/Assign training/Adam/Variable_58/read:02training/Adam/zeros_58:08
}
training/Adam/Variable_59:0 training/Adam/Variable_59/Assign training/Adam/Variable_59/read:02training/Adam/zeros_59:08
}
training/Adam/Variable_60:0 training/Adam/Variable_60/Assign training/Adam/Variable_60/read:02training/Adam/zeros_60:08
}
training/Adam/Variable_61:0 training/Adam/Variable_61/Assign training/Adam/Variable_61/read:02training/Adam/zeros_61:08
}
training/Adam/Variable_62:0 training/Adam/Variable_62/Assign training/Adam/Variable_62/read:02training/Adam/zeros_62:08
}
training/Adam/Variable_63:0 training/Adam/Variable_63/Assign training/Adam/Variable_63/read:02training/Adam/zeros_63:08
}
training/Adam/Variable_64:0 training/Adam/Variable_64/Assign training/Adam/Variable_64/read:02training/Adam/zeros_64:08
}
training/Adam/Variable_65:0 training/Adam/Variable_65/Assign training/Adam/Variable_65/read:02training/Adam/zeros_65:08
}
training/Adam/Variable_66:0 training/Adam/Variable_66/Assign training/Adam/Variable_66/read:02training/Adam/zeros_66:08
}
training/Adam/Variable_67:0 training/Adam/Variable_67/Assign training/Adam/Variable_67/read:02training/Adam/zeros_67:08
}
training/Adam/Variable_68:0 training/Adam/Variable_68/Assign training/Adam/Variable_68/read:02training/Adam/zeros_68:08
}
training/Adam/Variable_69:0 training/Adam/Variable_69/Assign training/Adam/Variable_69/read:02training/Adam/zeros_69:08
}
training/Adam/Variable_70:0 training/Adam/Variable_70/Assign training/Adam/Variable_70/read:02training/Adam/zeros_70:08
}
training/Adam/Variable_71:0 training/Adam/Variable_71/Assign training/Adam/Variable_71/read:02training/Adam/zeros_71:08
}
training/Adam/Variable_72:0 training/Adam/Variable_72/Assign training/Adam/Variable_72/read:02training/Adam/zeros_72:08
}
training/Adam/Variable_73:0 training/Adam/Variable_73/Assign training/Adam/Variable_73/read:02training/Adam/zeros_73:08
}
training/Adam/Variable_74:0 training/Adam/Variable_74/Assign training/Adam/Variable_74/read:02training/Adam/zeros_74:08
}
training/Adam/Variable_75:0 training/Adam/Variable_75/Assign training/Adam/Variable_75/read:02training/Adam/zeros_75:08
}
training/Adam/Variable_76:0 training/Adam/Variable_76/Assign training/Adam/Variable_76/read:02training/Adam/zeros_76:08
}
training/Adam/Variable_77:0 training/Adam/Variable_77/Assign training/Adam/Variable_77/read:02training/Adam/zeros_77:08
}
training/Adam/Variable_78:0 training/Adam/Variable_78/Assign training/Adam/Variable_78/read:02training/Adam/zeros_78:08
}
training/Adam/Variable_79:0 training/Adam/Variable_79/Assign training/Adam/Variable_79/read:02training/Adam/zeros_79:08
}
training/Adam/Variable_80:0 training/Adam/Variable_80/Assign training/Adam/Variable_80/read:02training/Adam/zeros_80:08
}
training/Adam/Variable_81:0 training/Adam/Variable_81/Assign training/Adam/Variable_81/read:02training/Adam/zeros_81:08
}
training/Adam/Variable_82:0 training/Adam/Variable_82/Assign training/Adam/Variable_82/read:02training/Adam/zeros_82:08
}
training/Adam/Variable_83:0 training/Adam/Variable_83/Assign training/Adam/Variable_83/read:02training/Adam/zeros_83:08
}
training/Adam/Variable_84:0 training/Adam/Variable_84/Assign training/Adam/Variable_84/read:02training/Adam/zeros_84:08
}
training/Adam/Variable_85:0 training/Adam/Variable_85/Assign training/Adam/Variable_85/read:02training/Adam/zeros_85:08
}
training/Adam/Variable_86:0 training/Adam/Variable_86/Assign training/Adam/Variable_86/read:02training/Adam/zeros_86:08
}
training/Adam/Variable_87:0 training/Adam/Variable_87/Assign training/Adam/Variable_87/read:02training/Adam/zeros_87:08
}
training/Adam/Variable_88:0 training/Adam/Variable_88/Assign training/Adam/Variable_88/read:02training/Adam/zeros_88:08
}
training/Adam/Variable_89:0 training/Adam/Variable_89/Assign training/Adam/Variable_89/read:02training/Adam/zeros_89:08
}
training/Adam/Variable_90:0 training/Adam/Variable_90/Assign training/Adam/Variable_90/read:02training/Adam/zeros_90:08
}
training/Adam/Variable_91:0 training/Adam/Variable_91/Assign training/Adam/Variable_91/read:02training/Adam/zeros_91:08
}
training/Adam/Variable_92:0 training/Adam/Variable_92/Assign training/Adam/Variable_92/read:02training/Adam/zeros_92:08
}
training/Adam/Variable_93:0 training/Adam/Variable_93/Assign training/Adam/Variable_93/read:02training/Adam/zeros_93:08
}
training/Adam/Variable_94:0 training/Adam/Variable_94/Assign training/Adam/Variable_94/read:02training/Adam/zeros_94:08
}
training/Adam/Variable_95:0 training/Adam/Variable_95/Assign training/Adam/Variable_95/read:02training/Adam/zeros_95:08
}
training/Adam/Variable_96:0 training/Adam/Variable_96/Assign training/Adam/Variable_96/read:02training/Adam/zeros_96:08
}
training/Adam/Variable_97:0 training/Adam/Variable_97/Assign training/Adam/Variable_97/read:02training/Adam/zeros_97:08
}
training/Adam/Variable_98:0 training/Adam/Variable_98/Assign training/Adam/Variable_98/read:02training/Adam/zeros_98:08
}
training/Adam/Variable_99:0 training/Adam/Variable_99/Assign training/Adam/Variable_99/read:02training/Adam/zeros_99:08

training/Adam/Variable_100:0!training/Adam/Variable_100/Assign!training/Adam/Variable_100/read:02training/Adam/zeros_100:08

training/Adam/Variable_101:0!training/Adam/Variable_101/Assign!training/Adam/Variable_101/read:02training/Adam/zeros_101:08

training/Adam/Variable_102:0!training/Adam/Variable_102/Assign!training/Adam/Variable_102/read:02training/Adam/zeros_102:08

training/Adam/Variable_103:0!training/Adam/Variable_103/Assign!training/Adam/Variable_103/read:02training/Adam/zeros_103:08

training/Adam/Variable_104:0!training/Adam/Variable_104/Assign!training/Adam/Variable_104/read:02training/Adam/zeros_104:08

training/Adam/Variable_105:0!training/Adam/Variable_105/Assign!training/Adam/Variable_105/read:02training/Adam/zeros_105:08

training/Adam/Variable_106:0!training/Adam/Variable_106/Assign!training/Adam/Variable_106/read:02training/Adam/zeros_106:08

training/Adam/Variable_107:0!training/Adam/Variable_107/Assign!training/Adam/Variable_107/read:02training/Adam/zeros_107:08

training/Adam/Variable_108:0!training/Adam/Variable_108/Assign!training/Adam/Variable_108/read:02training/Adam/zeros_108:08

training/Adam/Variable_109:0!training/Adam/Variable_109/Assign!training/Adam/Variable_109/read:02training/Adam/zeros_109:08

training/Adam/Variable_110:0!training/Adam/Variable_110/Assign!training/Adam/Variable_110/read:02training/Adam/zeros_110:08

training/Adam/Variable_111:0!training/Adam/Variable_111/Assign!training/Adam/Variable_111/read:02training/Adam/zeros_111:08

training/Adam/Variable_112:0!training/Adam/Variable_112/Assign!training/Adam/Variable_112/read:02training/Adam/zeros_112:08

training/Adam/Variable_113:0!training/Adam/Variable_113/Assign!training/Adam/Variable_113/read:02training/Adam/zeros_113:08

training/Adam/Variable_114:0!training/Adam/Variable_114/Assign!training/Adam/Variable_114/read:02training/Adam/zeros_114:08

training/Adam/Variable_115:0!training/Adam/Variable_115/Assign!training/Adam/Variable_115/read:02training/Adam/zeros_115:08

training/Adam/Variable_116:0!training/Adam/Variable_116/Assign!training/Adam/Variable_116/read:02training/Adam/zeros_116:08

training/Adam/Variable_117:0!training/Adam/Variable_117/Assign!training/Adam/Variable_117/read:02training/Adam/zeros_117:08

training/Adam/Variable_118:0!training/Adam/Variable_118/Assign!training/Adam/Variable_118/read:02training/Adam/zeros_118:08

training/Adam/Variable_119:0!training/Adam/Variable_119/Assign!training/Adam/Variable_119/read:02training/Adam/zeros_119:08

training/Adam/Variable_120:0!training/Adam/Variable_120/Assign!training/Adam/Variable_120/read:02training/Adam/zeros_120:08

training/Adam/Variable_121:0!training/Adam/Variable_121/Assign!training/Adam/Variable_121/read:02training/Adam/zeros_121:08

training/Adam/Variable_122:0!training/Adam/Variable_122/Assign!training/Adam/Variable_122/read:02training/Adam/zeros_122:08

training/Adam/Variable_123:0!training/Adam/Variable_123/Assign!training/Adam/Variable_123/read:02training/Adam/zeros_123:08

training/Adam/Variable_124:0!training/Adam/Variable_124/Assign!training/Adam/Variable_124/read:02training/Adam/zeros_124:08

training/Adam/Variable_125:0!training/Adam/Variable_125/Assign!training/Adam/Variable_125/read:02training/Adam/zeros_125:08

training/Adam/Variable_126:0!training/Adam/Variable_126/Assign!training/Adam/Variable_126/read:02training/Adam/zeros_126:08

training/Adam/Variable_127:0!training/Adam/Variable_127/Assign!training/Adam/Variable_127/read:02training/Adam/zeros_127:08

training/Adam/Variable_128:0!training/Adam/Variable_128/Assign!training/Adam/Variable_128/read:02training/Adam/zeros_128:08

training/Adam/Variable_129:0!training/Adam/Variable_129/Assign!training/Adam/Variable_129/read:02training/Adam/zeros_129:08

training/Adam/Variable_130:0!training/Adam/Variable_130/Assign!training/Adam/Variable_130/read:02training/Adam/zeros_130:08

training/Adam/Variable_131:0!training/Adam/Variable_131/Assign!training/Adam/Variable_131/read:02training/Adam/zeros_131:08

training/Adam/Variable_132:0!training/Adam/Variable_132/Assign!training/Adam/Variable_132/read:02training/Adam/zeros_132:08

training/Adam/Variable_133:0!training/Adam/Variable_133/Assign!training/Adam/Variable_133/read:02training/Adam/zeros_133:08

training/Adam/Variable_134:0!training/Adam/Variable_134/Assign!training/Adam/Variable_134/read:02training/Adam/zeros_134:08

training/Adam/Variable_135:0!training/Adam/Variable_135/Assign!training/Adam/Variable_135/read:02training/Adam/zeros_135:08

training/Adam/Variable_136:0!training/Adam/Variable_136/Assign!training/Adam/Variable_136/read:02training/Adam/zeros_136:08

training/Adam/Variable_137:0!training/Adam/Variable_137/Assign!training/Adam/Variable_137/read:02training/Adam/zeros_137:08č"ç       ŁK"	GčÇV$×A*

loss˛ü{>ki3&       ČÁ	ŤéÇV$×A*

val_lossŇ<s>=U%       Ř-	=ąV$×A*

lossWÚP><	Ü       ŮÜ2	RąV$×A*

val_loss<?>NěH       Ř-	ÉĆfV$×A*

loss\J>mlťŚ       ŮÜ2	ďÇfV$×A*

val_lossO˙>d<       Ř-	V$×A*

lossŚF>ćü       ŮÜ2	ßV$×A*

val_lossB>Ď9@       Ř-	fŃV$×A*

lossż-H>5Ś       ŮÜ2	gŃV$×A*

val_lossŹ>§źˇ       Ř-	ŮŚV$×A*

lossyšE>"eĎź       ŮÜ2	RŚV$×A*

val_lossSĘ>°V°|       Ř-	7úCŽV$×A*

lossż7>ŃQ       ŮÜ2	3űCŽV$×A*

val_lossXY>ÍÂa       Ř-	xC1śV$×A*

lossÔ<>|lQ       ŮÜ2	D1śV$×A*

val_lossR¸`>¨Őţ       Ř-	Đśä˝V$×A*

lossĹ>7>XSOŃ       ŮÜ2	ĺˇä˝V$×A*

val_lossAß6>§é÷       Ř-	íÖŐĹV$×A	*

lossYC>Y"­       ŮÜ2	ŘŐĹV$×A	*

val_losső3>xěJ/       Ř-	őÄÍV$×A
*

lossN6>ÚÁo	       ŮÜ2	bÄÍV$×A
*

val_lossV>+¨đ8       Ř-	§{ŐV$×A*

lossr5>7%Ć       ŮÜ2	É{ŐV$×A*

val_loss#{>ł7Ä*       Ř-	S	+ÝV$×A*

losse7>§l*\       ŮÜ2	é
+ÝV$×A*

val_loss¨`h>Fë]       Ř-	4ÓäV$×A*

loss?> 32       ŮÜ2	{ÓäV$×A*

val_loss\.>s       Ř-	?ĆěV$×A*

lossÎţ3>ţtM2       ŮÜ2	šÇěV$×A*

val_loss>Í>`ß       Ř-	<ôV$×A*

lossv,>&eSR       ŮÜ2	¨<ôV$×A*

val_loss*e>Ö"       Ř-	óűV$×A*

loss-W8>­K˝       ŮÜ2	đóűV$×A*

val_lossŮa>Ř VŁ       Ř-	ŚHŚW$×A*

loss3>CŚvĹ       ŮÜ2	ŞIŚW$×A*

val_lossmsv>°#­       Ř-	sUW$×A*

loss´l0>:s       ŮÜ2	UW$×A*

val_lossţö@>ĚÁA       Ř-	MW$×A*

lossbĺ1>ŠĽpĐ       ŮÜ2	MNW$×A*

val_lossÖO>őőu       Ř-	zťW$×A*

lossŽ,>       ŮÜ2	1{ťW$×A*

val_lossĚQs>SI       Ř-	Üq"W$×A*

lossßŹ->äXw       ŮÜ2	
q"W$×A*

val_lossYY>RyŃů       Ř-	Mf'*W$×A*

loss°82>e@Ň       ŮÜ2	wg'*W$×A*

val_lossbv>ţ