à
Û¬
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Ñ
|
dense_202/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_202/kernel
u
$dense_202/kernel/Read/ReadVariableOpReadVariableOpdense_202/kernel*
_output_shapes

:  *
dtype0
t
dense_202/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_202/bias
m
"dense_202/bias/Read/ReadVariableOpReadVariableOpdense_202/bias*
_output_shapes
: *
dtype0
|
dense_203/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense_203/kernel
u
$dense_203/kernel/Read/ReadVariableOpReadVariableOpdense_203/kernel*
_output_shapes

: *
dtype0
t
dense_203/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_203/bias
m
"dense_203/bias/Read/ReadVariableOpReadVariableOpdense_203/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

lstm_50/lstm_cell_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_namelstm_50/lstm_cell_50/kernel

/lstm_50/lstm_cell_50/kernel/Read/ReadVariableOpReadVariableOplstm_50/lstm_cell_50/kernel*
_output_shapes
:	*
dtype0
§
%lstm_50/lstm_cell_50/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *6
shared_name'%lstm_50/lstm_cell_50/recurrent_kernel
 
9lstm_50/lstm_cell_50/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_50/lstm_cell_50/recurrent_kernel*
_output_shapes
:	 *
dtype0

lstm_50/lstm_cell_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_50/lstm_cell_50/bias

-lstm_50/lstm_cell_50/bias/Read/ReadVariableOpReadVariableOplstm_50/lstm_cell_50/bias*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense_202/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_202/kernel/m

+Adam/dense_202/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_202/kernel/m*
_output_shapes

:  *
dtype0

Adam/dense_202/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_202/bias/m
{
)Adam/dense_202/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_202/bias/m*
_output_shapes
: *
dtype0

Adam/dense_203/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/dense_203/kernel/m

+Adam/dense_203/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_203/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_203/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_203/bias/m
{
)Adam/dense_203/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_203/bias/m*
_output_shapes
:*
dtype0
¡
"Adam/lstm_50/lstm_cell_50/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_50/lstm_cell_50/kernel/m

6Adam/lstm_50/lstm_cell_50/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_50/lstm_cell_50/kernel/m*
_output_shapes
:	*
dtype0
µ
,Adam/lstm_50/lstm_cell_50/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *=
shared_name.,Adam/lstm_50/lstm_cell_50/recurrent_kernel/m
®
@Adam/lstm_50/lstm_cell_50/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_50/lstm_cell_50/recurrent_kernel/m*
_output_shapes
:	 *
dtype0

 Adam/lstm_50/lstm_cell_50/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_50/lstm_cell_50/bias/m

4Adam/lstm_50/lstm_cell_50/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_50/lstm_cell_50/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_202/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_202/kernel/v

+Adam/dense_202/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_202/kernel/v*
_output_shapes

:  *
dtype0

Adam/dense_202/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_202/bias/v
{
)Adam/dense_202/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_202/bias/v*
_output_shapes
: *
dtype0

Adam/dense_203/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/dense_203/kernel/v

+Adam/dense_203/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_203/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_203/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_203/bias/v
{
)Adam/dense_203/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_203/bias/v*
_output_shapes
:*
dtype0
¡
"Adam/lstm_50/lstm_cell_50/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_50/lstm_cell_50/kernel/v

6Adam/lstm_50/lstm_cell_50/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_50/lstm_cell_50/kernel/v*
_output_shapes
:	*
dtype0
µ
,Adam/lstm_50/lstm_cell_50/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *=
shared_name.,Adam/lstm_50/lstm_cell_50/recurrent_kernel/v
®
@Adam/lstm_50/lstm_cell_50/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_50/lstm_cell_50/recurrent_kernel/v*
_output_shapes
:	 *
dtype0

 Adam/lstm_50/lstm_cell_50/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_50/lstm_cell_50/bias/v

4Adam/lstm_50/lstm_cell_50/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_50/lstm_cell_50/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ö2
valueÌ2BÉ2 BÂ2
Á
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
Á
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
Â
&iter

'beta_1

(beta_2
	)decay
*learning_ratemVmWmXmY+mZ,m[-m\v]v^v_v`+va,vb-vc*
5
+0
,1
-2
3
4
5
6*
5
+0
,1
-2
3
4
5
6*
* 
°
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

3serving_default* 
ã
4
state_size

+kernel
,recurrent_kernel
-bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9_random_generator
:__call__
*;&call_and_return_all_conditional_losses*
* 

+0
,1
-2*

+0
,1
-2*
* 


<states
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
`Z
VARIABLE_VALUEdense_202/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_202/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEdense_203/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_203/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_50/lstm_cell_50/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_50/lstm_cell_50/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_50/lstm_cell_50/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

L0*
* 
* 
* 
* 

+0
,1
-2*

+0
,1
-2*
* 

Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
5	variables
6trainable_variables
7regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	Rtotal
	Scount
T	variables
U	keras_api*
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

R0
S1*

T	variables*
}
VARIABLE_VALUEAdam/dense_202/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_202/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_203/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_203/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_50/lstm_cell_50/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_50/lstm_cell_50/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_50/lstm_cell_50/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_202/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_202/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_203/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_203/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_50/lstm_cell_50/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_50/lstm_cell_50/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_50/lstm_cell_50/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_lstm_50_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
é
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_50_inputlstm_50/lstm_cell_50/kernel%lstm_50/lstm_cell_50/recurrent_kernellstm_50/lstm_cell_50/biasdense_202/kerneldense_202/biasdense_203/kerneldense_203/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_962629
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_202/kernel/Read/ReadVariableOp"dense_202/bias/Read/ReadVariableOp$dense_203/kernel/Read/ReadVariableOp"dense_203/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_50/lstm_cell_50/kernel/Read/ReadVariableOp9lstm_50/lstm_cell_50/recurrent_kernel/Read/ReadVariableOp-lstm_50/lstm_cell_50/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_202/kernel/m/Read/ReadVariableOp)Adam/dense_202/bias/m/Read/ReadVariableOp+Adam/dense_203/kernel/m/Read/ReadVariableOp)Adam/dense_203/bias/m/Read/ReadVariableOp6Adam/lstm_50/lstm_cell_50/kernel/m/Read/ReadVariableOp@Adam/lstm_50/lstm_cell_50/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_50/lstm_cell_50/bias/m/Read/ReadVariableOp+Adam/dense_202/kernel/v/Read/ReadVariableOp)Adam/dense_202/bias/v/Read/ReadVariableOp+Adam/dense_203/kernel/v/Read/ReadVariableOp)Adam/dense_203/bias/v/Read/ReadVariableOp6Adam/lstm_50/lstm_cell_50/kernel/v/Read/ReadVariableOp@Adam/lstm_50/lstm_cell_50/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_50/lstm_cell_50/bias/v/Read/ReadVariableOpConst*)
Tin"
 2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_963489
Ð
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_202/kerneldense_202/biasdense_203/kerneldense_203/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_50/lstm_cell_50/kernel%lstm_50/lstm_cell_50/recurrent_kernellstm_50/lstm_cell_50/biastotalcountAdam/dense_202/kernel/mAdam/dense_202/bias/mAdam/dense_203/kernel/mAdam/dense_203/bias/m"Adam/lstm_50/lstm_cell_50/kernel/m,Adam/lstm_50/lstm_cell_50/recurrent_kernel/m Adam/lstm_50/lstm_cell_50/bias/mAdam/dense_202/kernel/vAdam/dense_202/bias/vAdam/dense_203/kernel/vAdam/dense_203/bias/v"Adam/lstm_50/lstm_cell_50/kernel/v,Adam/lstm_50/lstm_cell_50/recurrent_kernel/v Adam/lstm_50/lstm_cell_50/bias/v*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_963583ØÊ
Ò
À
J__inference_sequential_101_layer_call_and_return_conditional_losses_961930

inputs!
lstm_50_961889:	!
lstm_50_961891:	 
lstm_50_961893:	"
dense_202_961908:  
dense_202_961910: "
dense_203_961924: 
dense_203_961926:
identity¢!dense_202/StatefulPartitionedCall¢!dense_203/StatefulPartitionedCall¢lstm_50/StatefulPartitionedCallþ
lstm_50/StatefulPartitionedCallStatefulPartitionedCallinputslstm_50_961889lstm_50_961891lstm_50_961893*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_50_layer_call_and_return_conditional_losses_961888
!dense_202/StatefulPartitionedCallStatefulPartitionedCall(lstm_50/StatefulPartitionedCall:output:0dense_202_961908dense_202_961910*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_202_layer_call_and_return_conditional_losses_961907
!dense_203/StatefulPartitionedCallStatefulPartitionedCall*dense_202/StatefulPartitionedCall:output:0dense_203_961924dense_203_961926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_203_layer_call_and_return_conditional_losses_961923y
IdentityIdentity*dense_203/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
NoOpNoOp"^dense_202/StatefulPartitionedCall"^dense_203/StatefulPartitionedCall ^lstm_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : : : : 2F
!dense_202/StatefulPartitionedCall!dense_202/StatefulPartitionedCall2F
!dense_203/StatefulPartitionedCall!dense_203/StatefulPartitionedCall2B
lstm_50/StatefulPartitionedCalllstm_50/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
Ã
while_cond_963017
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_963017___redundant_placeholder04
0while_while_cond_963017___redundant_placeholder14
0while_while_cond_963017___redundant_placeholder24
0while_while_cond_963017___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Ò
À
J__inference_sequential_101_layer_call_and_return_conditional_losses_962174

inputs!
lstm_50_962156:	!
lstm_50_962158:	 
lstm_50_962160:	"
dense_202_962163:  
dense_202_962165: "
dense_203_962168: 
dense_203_962170:
identity¢!dense_202/StatefulPartitionedCall¢!dense_203/StatefulPartitionedCall¢lstm_50/StatefulPartitionedCallþ
lstm_50/StatefulPartitionedCallStatefulPartitionedCallinputslstm_50_962156lstm_50_962158lstm_50_962160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_50_layer_call_and_return_conditional_losses_962123
!dense_202/StatefulPartitionedCallStatefulPartitionedCall(lstm_50/StatefulPartitionedCall:output:0dense_202_962163dense_202_962165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_202_layer_call_and_return_conditional_losses_961907
!dense_203/StatefulPartitionedCallStatefulPartitionedCall*dense_202/StatefulPartitionedCall:output:0dense_203_962168dense_203_962170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_203_layer_call_and_return_conditional_losses_961923y
IdentityIdentity*dense_203/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
NoOpNoOp"^dense_202/StatefulPartitionedCall"^dense_203/StatefulPartitionedCall ^lstm_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : : : : 2F
!dense_202/StatefulPartitionedCall!dense_202/StatefulPartitionedCall2F
!dense_203/StatefulPartitionedCall!dense_203/StatefulPartitionedCall2B
lstm_50/StatefulPartitionedCalllstm_50/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý

H__inference_lstm_cell_50_layer_call_and_return_conditional_losses_963382

inputs
states_0
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
¨J

C__inference_lstm_50_layer_call_and_return_conditional_losses_963102

inputs>
+lstm_cell_50_matmul_readvariableop_resource:	@
-lstm_cell_50_matmul_1_readvariableop_resource:	 ;
,lstm_cell_50_biasadd_readvariableop_resource:	
identity¢#lstm_cell_50/BiasAdd/ReadVariableOp¢"lstm_cell_50/MatMul/ReadVariableOp¢$lstm_cell_50/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_50/MatMul/ReadVariableOpReadVariableOp+lstm_cell_50_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_50/MatMulMatMulstrided_slice_2:output:0*lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_50_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_50/MatMul_1MatMulzeros:output:0,lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_50/addAddV2lstm_cell_50/MatMul:product:0lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_50_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_50/BiasAddBiasAddlstm_cell_50/add:z:0+lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_50/splitSplit%lstm_cell_50/split/split_dim:output:0lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_50/SigmoidSigmoidlstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_50/Sigmoid_1Sigmoidlstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_50/mulMullstm_cell_50/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_50/ReluRelulstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_50/mul_1Mullstm_cell_50/Sigmoid:y:0lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_50/add_1AddV2lstm_cell_50/mul:z:0lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_50/Sigmoid_2Sigmoidlstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_50/Relu_1Relulstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_50/mul_2Mullstm_cell_50/Sigmoid_2:y:0!lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_50_matmul_readvariableop_resource-lstm_cell_50_matmul_1_readvariableop_resource,lstm_cell_50_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_963018*
condR
while_cond_963017*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_50/BiasAdd/ReadVariableOp#^lstm_cell_50/MatMul/ReadVariableOp%^lstm_cell_50/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_50/BiasAdd/ReadVariableOp#lstm_cell_50/BiasAdd/ReadVariableOp2H
"lstm_cell_50/MatMul/ReadVariableOp"lstm_cell_50/MatMul/ReadVariableOp2L
$lstm_cell_50/MatMul_1/ReadVariableOp$lstm_cell_50/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ö
E__inference_dense_202_layer_call_and_return_conditional_losses_963265

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ä

*__inference_dense_202_layer_call_fn_963254

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_202_layer_call_and_return_conditional_losses_961907o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÝJ

C__inference_lstm_50_layer_call_and_return_conditional_losses_962816
inputs_0>
+lstm_cell_50_matmul_readvariableop_resource:	@
-lstm_cell_50_matmul_1_readvariableop_resource:	 ;
,lstm_cell_50_biasadd_readvariableop_resource:	
identity¢#lstm_cell_50/BiasAdd/ReadVariableOp¢"lstm_cell_50/MatMul/ReadVariableOp¢$lstm_cell_50/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_50/MatMul/ReadVariableOpReadVariableOp+lstm_cell_50_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_50/MatMulMatMulstrided_slice_2:output:0*lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_50_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_50/MatMul_1MatMulzeros:output:0,lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_50/addAddV2lstm_cell_50/MatMul:product:0lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_50_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_50/BiasAddBiasAddlstm_cell_50/add:z:0+lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_50/splitSplit%lstm_cell_50/split/split_dim:output:0lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_50/SigmoidSigmoidlstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_50/Sigmoid_1Sigmoidlstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_50/mulMullstm_cell_50/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_50/ReluRelulstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_50/mul_1Mullstm_cell_50/Sigmoid:y:0lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_50/add_1AddV2lstm_cell_50/mul:z:0lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_50/Sigmoid_2Sigmoidlstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_50/Relu_1Relulstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_50/mul_2Mullstm_cell_50/Sigmoid_2:y:0!lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_50_matmul_readvariableop_resource-lstm_cell_50_matmul_1_readvariableop_resource,lstm_cell_50_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_962732*
condR
while_cond_962731*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_50/BiasAdd/ReadVariableOp#^lstm_cell_50/MatMul/ReadVariableOp%^lstm_cell_50/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_50/BiasAdd/ReadVariableOp#lstm_cell_50/BiasAdd/ReadVariableOp2H
"lstm_cell_50/MatMul/ReadVariableOp"lstm_cell_50/MatMul/ReadVariableOp2L
$lstm_cell_50/MatMul_1/ReadVariableOp$lstm_cell_50/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Õ

H__inference_lstm_cell_50_layer_call_and_return_conditional_losses_961456

inputs

states
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
ë
ö
-__inference_lstm_cell_50_layer_call_fn_963318

inputs
states_0
states_1
unknown:	
	unknown_0:	 
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_50_layer_call_and_return_conditional_losses_961602o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
d
Î
J__inference_sequential_101_layer_call_and_return_conditional_losses_962452

inputsF
3lstm_50_lstm_cell_50_matmul_readvariableop_resource:	H
5lstm_50_lstm_cell_50_matmul_1_readvariableop_resource:	 C
4lstm_50_lstm_cell_50_biasadd_readvariableop_resource:	:
(dense_202_matmul_readvariableop_resource:  7
)dense_202_biasadd_readvariableop_resource: :
(dense_203_matmul_readvariableop_resource: 7
)dense_203_biasadd_readvariableop_resource:
identity¢ dense_202/BiasAdd/ReadVariableOp¢dense_202/MatMul/ReadVariableOp¢ dense_203/BiasAdd/ReadVariableOp¢dense_203/MatMul/ReadVariableOp¢+lstm_50/lstm_cell_50/BiasAdd/ReadVariableOp¢*lstm_50/lstm_cell_50/MatMul/ReadVariableOp¢,lstm_50/lstm_cell_50/MatMul_1/ReadVariableOp¢lstm_50/whileC
lstm_50/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_50/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_50/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_50/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_50/strided_sliceStridedSlicelstm_50/Shape:output:0$lstm_50/strided_slice/stack:output:0&lstm_50/strided_slice/stack_1:output:0&lstm_50/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_50/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_50/zeros/packedPacklstm_50/strided_slice:output:0lstm_50/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_50/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_50/zerosFilllstm_50/zeros/packed:output:0lstm_50/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
lstm_50/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_50/zeros_1/packedPacklstm_50/strided_slice:output:0!lstm_50/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_50/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_50/zeros_1Filllstm_50/zeros_1/packed:output:0lstm_50/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
lstm_50/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_50/transpose	Transposeinputslstm_50/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
lstm_50/Shape_1Shapelstm_50/transpose:y:0*
T0*
_output_shapes
:g
lstm_50/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_50/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_50/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_50/strided_slice_1StridedSlicelstm_50/Shape_1:output:0&lstm_50/strided_slice_1/stack:output:0(lstm_50/strided_slice_1/stack_1:output:0(lstm_50/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_50/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_50/TensorArrayV2TensorListReserve,lstm_50/TensorArrayV2/element_shape:output:0 lstm_50/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_50/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_50/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_50/transpose:y:0Flstm_50/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_50/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_50/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_50/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_50/strided_slice_2StridedSlicelstm_50/transpose:y:0&lstm_50/strided_slice_2/stack:output:0(lstm_50/strided_slice_2/stack_1:output:0(lstm_50/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstm_50/lstm_cell_50/MatMul/ReadVariableOpReadVariableOp3lstm_50_lstm_cell_50_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0®
lstm_50/lstm_cell_50/MatMulMatMul lstm_50/strided_slice_2:output:02lstm_50/lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,lstm_50/lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp5lstm_50_lstm_cell_50_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0¨
lstm_50/lstm_cell_50/MatMul_1MatMullstm_50/zeros:output:04lstm_50/lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm_50/lstm_cell_50/addAddV2%lstm_50/lstm_cell_50/MatMul:product:0'lstm_50/lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstm_50/lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp4lstm_50_lstm_cell_50_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstm_50/lstm_cell_50/BiasAddBiasAddlstm_50/lstm_cell_50/add:z:03lstm_50/lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstm_50/lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :õ
lstm_50/lstm_cell_50/splitSplit-lstm_50/lstm_cell_50/split/split_dim:output:0%lstm_50/lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split~
lstm_50/lstm_cell_50/SigmoidSigmoid#lstm_50/lstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/lstm_cell_50/Sigmoid_1Sigmoid#lstm_50/lstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/lstm_cell_50/mulMul"lstm_50/lstm_cell_50/Sigmoid_1:y:0lstm_50/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
lstm_50/lstm_cell_50/ReluRelu#lstm_50/lstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/lstm_cell_50/mul_1Mul lstm_50/lstm_cell_50/Sigmoid:y:0'lstm_50/lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/lstm_cell_50/add_1AddV2lstm_50/lstm_cell_50/mul:z:0lstm_50/lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/lstm_cell_50/Sigmoid_2Sigmoid#lstm_50/lstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
lstm_50/lstm_cell_50/Relu_1Relulstm_50/lstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¢
lstm_50/lstm_cell_50/mul_2Mul"lstm_50/lstm_cell_50/Sigmoid_2:y:0)lstm_50/lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
%lstm_50/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ð
lstm_50/TensorArrayV2_1TensorListReserve.lstm_50/TensorArrayV2_1/element_shape:output:0 lstm_50/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_50/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_50/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_50/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ò
lstm_50/whileWhile#lstm_50/while/loop_counter:output:0)lstm_50/while/maximum_iterations:output:0lstm_50/time:output:0 lstm_50/TensorArrayV2_1:handle:0lstm_50/zeros:output:0lstm_50/zeros_1:output:0 lstm_50/strided_slice_1:output:0?lstm_50/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_50_lstm_cell_50_matmul_readvariableop_resource5lstm_50_lstm_cell_50_matmul_1_readvariableop_resource4lstm_50_lstm_cell_50_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_50_while_body_962355*%
condR
lstm_50_while_cond_962354*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
8lstm_50/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ú
*lstm_50/TensorArrayV2Stack/TensorListStackTensorListStacklstm_50/while:output:3Alstm_50/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0p
lstm_50/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_50/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_50/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_50/strided_slice_3StridedSlice3lstm_50/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_50/strided_slice_3/stack:output:0(lstm_50/strided_slice_3/stack_1:output:0(lstm_50/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
lstm_50/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_50/transpose_1	Transpose3lstm_50/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_50/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
lstm_50/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_202/MatMul/ReadVariableOpReadVariableOp(dense_202_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_202/MatMulMatMul lstm_50/strided_slice_3:output:0'dense_202/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 dense_202/BiasAdd/ReadVariableOpReadVariableOp)dense_202_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_202/BiasAddBiasAdddense_202/MatMul:product:0(dense_202/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
dense_202/ReluReludense_202/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_203/MatMul/ReadVariableOpReadVariableOp(dense_203_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_203/MatMulMatMuldense_202/Relu:activations:0'dense_203/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_203/BiasAdd/ReadVariableOpReadVariableOp)dense_203_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_203/BiasAddBiasAdddense_203/MatMul:product:0(dense_203/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_203/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp!^dense_202/BiasAdd/ReadVariableOp ^dense_202/MatMul/ReadVariableOp!^dense_203/BiasAdd/ReadVariableOp ^dense_203/MatMul/ReadVariableOp,^lstm_50/lstm_cell_50/BiasAdd/ReadVariableOp+^lstm_50/lstm_cell_50/MatMul/ReadVariableOp-^lstm_50/lstm_cell_50/MatMul_1/ReadVariableOp^lstm_50/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : : : : 2D
 dense_202/BiasAdd/ReadVariableOp dense_202/BiasAdd/ReadVariableOp2B
dense_202/MatMul/ReadVariableOpdense_202/MatMul/ReadVariableOp2D
 dense_203/BiasAdd/ReadVariableOp dense_203/BiasAdd/ReadVariableOp2B
dense_203/MatMul/ReadVariableOpdense_203/MatMul/ReadVariableOp2Z
+lstm_50/lstm_cell_50/BiasAdd/ReadVariableOp+lstm_50/lstm_cell_50/BiasAdd/ReadVariableOp2X
*lstm_50/lstm_cell_50/MatMul/ReadVariableOp*lstm_50/lstm_cell_50/MatMul/ReadVariableOp2\
,lstm_50/lstm_cell_50/MatMul_1/ReadVariableOp,lstm_50/lstm_cell_50/MatMul_1/ReadVariableOp2
lstm_50/whilelstm_50/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
8
Ð
while_body_962875
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_50_matmul_readvariableop_resource_0:	H
5while_lstm_cell_50_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_50_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_50_matmul_readvariableop_resource:	F
3while_lstm_cell_50_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_50_biasadd_readvariableop_resource:	¢)while/lstm_cell_50/BiasAdd/ReadVariableOp¢(while/lstm_cell_50/MatMul/ReadVariableOp¢*while/lstm_cell_50/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_50/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_50_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_50/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_50/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_50/addAddV2#while/lstm_cell_50/MatMul:product:0%while/lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_50_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_50/BiasAddBiasAddwhile/lstm_cell_50/add:z:01while/lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_50/splitSplit+while/lstm_cell_50/split/split_dim:output:0#while/lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_50/SigmoidSigmoid!while/lstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_50/Sigmoid_1Sigmoid!while/lstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mulMul while/lstm_cell_50/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_50/ReluRelu!while/lstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mul_1Mulwhile/lstm_cell_50/Sigmoid:y:0%while/lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/add_1AddV2while/lstm_cell_50/mul:z:0while/lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_50/Sigmoid_2Sigmoid!while/lstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_50/Relu_1Reluwhile/lstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mul_2Mul while/lstm_cell_50/Sigmoid_2:y:0'while/lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_50/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_50/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_50/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_50/BiasAdd/ReadVariableOp)^while/lstm_cell_50/MatMul/ReadVariableOp+^while/lstm_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_50_biasadd_readvariableop_resource4while_lstm_cell_50_biasadd_readvariableop_resource_0"l
3while_lstm_cell_50_matmul_1_readvariableop_resource5while_lstm_cell_50_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_50_matmul_readvariableop_resource3while_lstm_cell_50_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_50/BiasAdd/ReadVariableOp)while/lstm_cell_50/BiasAdd/ReadVariableOp2T
(while/lstm_cell_50/MatMul/ReadVariableOp(while/lstm_cell_50/MatMul/ReadVariableOp2X
*while/lstm_cell_50/MatMul_1/ReadVariableOp*while/lstm_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ØA
Ð

lstm_50_while_body_962355,
(lstm_50_while_lstm_50_while_loop_counter2
.lstm_50_while_lstm_50_while_maximum_iterations
lstm_50_while_placeholder
lstm_50_while_placeholder_1
lstm_50_while_placeholder_2
lstm_50_while_placeholder_3+
'lstm_50_while_lstm_50_strided_slice_1_0g
clstm_50_while_tensorarrayv2read_tensorlistgetitem_lstm_50_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_50_while_lstm_cell_50_matmul_readvariableop_resource_0:	P
=lstm_50_while_lstm_cell_50_matmul_1_readvariableop_resource_0:	 K
<lstm_50_while_lstm_cell_50_biasadd_readvariableop_resource_0:	
lstm_50_while_identity
lstm_50_while_identity_1
lstm_50_while_identity_2
lstm_50_while_identity_3
lstm_50_while_identity_4
lstm_50_while_identity_5)
%lstm_50_while_lstm_50_strided_slice_1e
alstm_50_while_tensorarrayv2read_tensorlistgetitem_lstm_50_tensorarrayunstack_tensorlistfromtensorL
9lstm_50_while_lstm_cell_50_matmul_readvariableop_resource:	N
;lstm_50_while_lstm_cell_50_matmul_1_readvariableop_resource:	 I
:lstm_50_while_lstm_cell_50_biasadd_readvariableop_resource:	¢1lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOp¢0lstm_50/while/lstm_cell_50/MatMul/ReadVariableOp¢2lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOp
?lstm_50/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_50/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_50_while_tensorarrayv2read_tensorlistgetitem_lstm_50_tensorarrayunstack_tensorlistfromtensor_0lstm_50_while_placeholderHlstm_50/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0­
0lstm_50/while/lstm_cell_50/MatMul/ReadVariableOpReadVariableOp;lstm_50_while_lstm_cell_50_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ò
!lstm_50/while/lstm_cell_50/MatMulMatMul8lstm_50/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_50/while/lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
2lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp=lstm_50_while_lstm_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¹
#lstm_50/while/lstm_cell_50/MatMul_1MatMullstm_50_while_placeholder_2:lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm_50/while/lstm_cell_50/addAddV2+lstm_50/while/lstm_cell_50/MatMul:product:0-lstm_50/while/lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp<lstm_50_while_lstm_cell_50_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstm_50/while/lstm_cell_50/BiasAddBiasAdd"lstm_50/while/lstm_cell_50/add:z:09lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstm_50/while/lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_50/while/lstm_cell_50/splitSplit3lstm_50/while/lstm_cell_50/split/split_dim:output:0+lstm_50/while/lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
"lstm_50/while/lstm_cell_50/SigmoidSigmoid)lstm_50/while/lstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
$lstm_50/while/lstm_cell_50/Sigmoid_1Sigmoid)lstm_50/while/lstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/while/lstm_cell_50/mulMul(lstm_50/while/lstm_cell_50/Sigmoid_1:y:0lstm_50_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/while/lstm_cell_50/ReluRelu)lstm_50/while/lstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
 lstm_50/while/lstm_cell_50/mul_1Mul&lstm_50/while/lstm_cell_50/Sigmoid:y:0-lstm_50/while/lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
 lstm_50/while/lstm_cell_50/add_1AddV2"lstm_50/while/lstm_cell_50/mul:z:0$lstm_50/while/lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
$lstm_50/while/lstm_cell_50/Sigmoid_2Sigmoid)lstm_50/while/lstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!lstm_50/while/lstm_cell_50/Relu_1Relu$lstm_50/while/lstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ´
 lstm_50/while/lstm_cell_50/mul_2Mul(lstm_50/while/lstm_cell_50/Sigmoid_2:y:0/lstm_50/while/lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ å
2lstm_50/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_50_while_placeholder_1lstm_50_while_placeholder$lstm_50/while/lstm_cell_50/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_50/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_50/while/addAddV2lstm_50_while_placeholderlstm_50/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_50/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_50/while/add_1AddV2(lstm_50_while_lstm_50_while_loop_counterlstm_50/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_50/while/IdentityIdentitylstm_50/while/add_1:z:0^lstm_50/while/NoOp*
T0*
_output_shapes
: 
lstm_50/while/Identity_1Identity.lstm_50_while_lstm_50_while_maximum_iterations^lstm_50/while/NoOp*
T0*
_output_shapes
: q
lstm_50/while/Identity_2Identitylstm_50/while/add:z:0^lstm_50/while/NoOp*
T0*
_output_shapes
: ±
lstm_50/while/Identity_3IdentityBlstm_50/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_50/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_50/while/Identity_4Identity$lstm_50/while/lstm_cell_50/mul_2:z:0^lstm_50/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/while/Identity_5Identity$lstm_50/while/lstm_cell_50/add_1:z:0^lstm_50/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð
lstm_50/while/NoOpNoOp2^lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOp1^lstm_50/while/lstm_cell_50/MatMul/ReadVariableOp3^lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_50_while_identitylstm_50/while/Identity:output:0"=
lstm_50_while_identity_1!lstm_50/while/Identity_1:output:0"=
lstm_50_while_identity_2!lstm_50/while/Identity_2:output:0"=
lstm_50_while_identity_3!lstm_50/while/Identity_3:output:0"=
lstm_50_while_identity_4!lstm_50/while/Identity_4:output:0"=
lstm_50_while_identity_5!lstm_50/while/Identity_5:output:0"P
%lstm_50_while_lstm_50_strided_slice_1'lstm_50_while_lstm_50_strided_slice_1_0"z
:lstm_50_while_lstm_cell_50_biasadd_readvariableop_resource<lstm_50_while_lstm_cell_50_biasadd_readvariableop_resource_0"|
;lstm_50_while_lstm_cell_50_matmul_1_readvariableop_resource=lstm_50_while_lstm_cell_50_matmul_1_readvariableop_resource_0"x
9lstm_50_while_lstm_cell_50_matmul_readvariableop_resource;lstm_50_while_lstm_cell_50_matmul_readvariableop_resource_0"È
alstm_50_while_tensorarrayv2read_tensorlistgetitem_lstm_50_tensorarrayunstack_tensorlistfromtensorclstm_50_while_tensorarrayv2read_tensorlistgetitem_lstm_50_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2f
1lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOp1lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOp2d
0lstm_50/while/lstm_cell_50/MatMul/ReadVariableOp0lstm_50/while/lstm_cell_50/MatMul/ReadVariableOp2h
2lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOp2lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
8
Ð
while_body_961804
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_50_matmul_readvariableop_resource_0:	H
5while_lstm_cell_50_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_50_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_50_matmul_readvariableop_resource:	F
3while_lstm_cell_50_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_50_biasadd_readvariableop_resource:	¢)while/lstm_cell_50/BiasAdd/ReadVariableOp¢(while/lstm_cell_50/MatMul/ReadVariableOp¢*while/lstm_cell_50/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_50/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_50_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_50/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_50/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_50/addAddV2#while/lstm_cell_50/MatMul:product:0%while/lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_50_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_50/BiasAddBiasAddwhile/lstm_cell_50/add:z:01while/lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_50/splitSplit+while/lstm_cell_50/split/split_dim:output:0#while/lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_50/SigmoidSigmoid!while/lstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_50/Sigmoid_1Sigmoid!while/lstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mulMul while/lstm_cell_50/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_50/ReluRelu!while/lstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mul_1Mulwhile/lstm_cell_50/Sigmoid:y:0%while/lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/add_1AddV2while/lstm_cell_50/mul:z:0while/lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_50/Sigmoid_2Sigmoid!while/lstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_50/Relu_1Reluwhile/lstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mul_2Mul while/lstm_cell_50/Sigmoid_2:y:0'while/lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_50/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_50/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_50/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_50/BiasAdd/ReadVariableOp)^while/lstm_cell_50/MatMul/ReadVariableOp+^while/lstm_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_50_biasadd_readvariableop_resource4while_lstm_cell_50_biasadd_readvariableop_resource_0"l
3while_lstm_cell_50_matmul_1_readvariableop_resource5while_lstm_cell_50_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_50_matmul_readvariableop_resource3while_lstm_cell_50_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_50/BiasAdd/ReadVariableOp)while/lstm_cell_50/BiasAdd/ReadVariableOp2T
(while/lstm_cell_50/MatMul/ReadVariableOp(while/lstm_cell_50/MatMul/ReadVariableOp2X
*while/lstm_cell_50/MatMul_1/ReadVariableOp*while/lstm_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
µ
Ã
while_cond_963160
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_963160___redundant_placeholder04
0while_while_cond_963160___redundant_placeholder14
0while_while_cond_963160___redundant_placeholder24
0while_while_cond_963160___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ØA
Ð

lstm_50_while_body_962511,
(lstm_50_while_lstm_50_while_loop_counter2
.lstm_50_while_lstm_50_while_maximum_iterations
lstm_50_while_placeholder
lstm_50_while_placeholder_1
lstm_50_while_placeholder_2
lstm_50_while_placeholder_3+
'lstm_50_while_lstm_50_strided_slice_1_0g
clstm_50_while_tensorarrayv2read_tensorlistgetitem_lstm_50_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_50_while_lstm_cell_50_matmul_readvariableop_resource_0:	P
=lstm_50_while_lstm_cell_50_matmul_1_readvariableop_resource_0:	 K
<lstm_50_while_lstm_cell_50_biasadd_readvariableop_resource_0:	
lstm_50_while_identity
lstm_50_while_identity_1
lstm_50_while_identity_2
lstm_50_while_identity_3
lstm_50_while_identity_4
lstm_50_while_identity_5)
%lstm_50_while_lstm_50_strided_slice_1e
alstm_50_while_tensorarrayv2read_tensorlistgetitem_lstm_50_tensorarrayunstack_tensorlistfromtensorL
9lstm_50_while_lstm_cell_50_matmul_readvariableop_resource:	N
;lstm_50_while_lstm_cell_50_matmul_1_readvariableop_resource:	 I
:lstm_50_while_lstm_cell_50_biasadd_readvariableop_resource:	¢1lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOp¢0lstm_50/while/lstm_cell_50/MatMul/ReadVariableOp¢2lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOp
?lstm_50/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_50/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_50_while_tensorarrayv2read_tensorlistgetitem_lstm_50_tensorarrayunstack_tensorlistfromtensor_0lstm_50_while_placeholderHlstm_50/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0­
0lstm_50/while/lstm_cell_50/MatMul/ReadVariableOpReadVariableOp;lstm_50_while_lstm_cell_50_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ò
!lstm_50/while/lstm_cell_50/MatMulMatMul8lstm_50/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_50/while/lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
2lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp=lstm_50_while_lstm_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¹
#lstm_50/while/lstm_cell_50/MatMul_1MatMullstm_50_while_placeholder_2:lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm_50/while/lstm_cell_50/addAddV2+lstm_50/while/lstm_cell_50/MatMul:product:0-lstm_50/while/lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp<lstm_50_while_lstm_cell_50_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstm_50/while/lstm_cell_50/BiasAddBiasAdd"lstm_50/while/lstm_cell_50/add:z:09lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstm_50/while/lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_50/while/lstm_cell_50/splitSplit3lstm_50/while/lstm_cell_50/split/split_dim:output:0+lstm_50/while/lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
"lstm_50/while/lstm_cell_50/SigmoidSigmoid)lstm_50/while/lstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
$lstm_50/while/lstm_cell_50/Sigmoid_1Sigmoid)lstm_50/while/lstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/while/lstm_cell_50/mulMul(lstm_50/while/lstm_cell_50/Sigmoid_1:y:0lstm_50_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/while/lstm_cell_50/ReluRelu)lstm_50/while/lstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
 lstm_50/while/lstm_cell_50/mul_1Mul&lstm_50/while/lstm_cell_50/Sigmoid:y:0-lstm_50/while/lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
 lstm_50/while/lstm_cell_50/add_1AddV2"lstm_50/while/lstm_cell_50/mul:z:0$lstm_50/while/lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
$lstm_50/while/lstm_cell_50/Sigmoid_2Sigmoid)lstm_50/while/lstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!lstm_50/while/lstm_cell_50/Relu_1Relu$lstm_50/while/lstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ´
 lstm_50/while/lstm_cell_50/mul_2Mul(lstm_50/while/lstm_cell_50/Sigmoid_2:y:0/lstm_50/while/lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ å
2lstm_50/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_50_while_placeholder_1lstm_50_while_placeholder$lstm_50/while/lstm_cell_50/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_50/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_50/while/addAddV2lstm_50_while_placeholderlstm_50/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_50/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_50/while/add_1AddV2(lstm_50_while_lstm_50_while_loop_counterlstm_50/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_50/while/IdentityIdentitylstm_50/while/add_1:z:0^lstm_50/while/NoOp*
T0*
_output_shapes
: 
lstm_50/while/Identity_1Identity.lstm_50_while_lstm_50_while_maximum_iterations^lstm_50/while/NoOp*
T0*
_output_shapes
: q
lstm_50/while/Identity_2Identitylstm_50/while/add:z:0^lstm_50/while/NoOp*
T0*
_output_shapes
: ±
lstm_50/while/Identity_3IdentityBlstm_50/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_50/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_50/while/Identity_4Identity$lstm_50/while/lstm_cell_50/mul_2:z:0^lstm_50/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/while/Identity_5Identity$lstm_50/while/lstm_cell_50/add_1:z:0^lstm_50/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð
lstm_50/while/NoOpNoOp2^lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOp1^lstm_50/while/lstm_cell_50/MatMul/ReadVariableOp3^lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_50_while_identitylstm_50/while/Identity:output:0"=
lstm_50_while_identity_1!lstm_50/while/Identity_1:output:0"=
lstm_50_while_identity_2!lstm_50/while/Identity_2:output:0"=
lstm_50_while_identity_3!lstm_50/while/Identity_3:output:0"=
lstm_50_while_identity_4!lstm_50/while/Identity_4:output:0"=
lstm_50_while_identity_5!lstm_50/while/Identity_5:output:0"P
%lstm_50_while_lstm_50_strided_slice_1'lstm_50_while_lstm_50_strided_slice_1_0"z
:lstm_50_while_lstm_cell_50_biasadd_readvariableop_resource<lstm_50_while_lstm_cell_50_biasadd_readvariableop_resource_0"|
;lstm_50_while_lstm_cell_50_matmul_1_readvariableop_resource=lstm_50_while_lstm_cell_50_matmul_1_readvariableop_resource_0"x
9lstm_50_while_lstm_cell_50_matmul_readvariableop_resource;lstm_50_while_lstm_cell_50_matmul_readvariableop_resource_0"È
alstm_50_while_tensorarrayv2read_tensorlistgetitem_lstm_50_tensorarrayunstack_tensorlistfromtensorclstm_50_while_tensorarrayv2read_tensorlistgetitem_lstm_50_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2f
1lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOp1lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOp2d
0lstm_50/while/lstm_cell_50/MatMul/ReadVariableOp0lstm_50/while/lstm_cell_50/MatMul/ReadVariableOp2h
2lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOp2lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
8
Ð
while_body_963018
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_50_matmul_readvariableop_resource_0:	H
5while_lstm_cell_50_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_50_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_50_matmul_readvariableop_resource:	F
3while_lstm_cell_50_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_50_biasadd_readvariableop_resource:	¢)while/lstm_cell_50/BiasAdd/ReadVariableOp¢(while/lstm_cell_50/MatMul/ReadVariableOp¢*while/lstm_cell_50/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_50/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_50_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_50/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_50/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_50/addAddV2#while/lstm_cell_50/MatMul:product:0%while/lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_50_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_50/BiasAddBiasAddwhile/lstm_cell_50/add:z:01while/lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_50/splitSplit+while/lstm_cell_50/split/split_dim:output:0#while/lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_50/SigmoidSigmoid!while/lstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_50/Sigmoid_1Sigmoid!while/lstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mulMul while/lstm_cell_50/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_50/ReluRelu!while/lstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mul_1Mulwhile/lstm_cell_50/Sigmoid:y:0%while/lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/add_1AddV2while/lstm_cell_50/mul:z:0while/lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_50/Sigmoid_2Sigmoid!while/lstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_50/Relu_1Reluwhile/lstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mul_2Mul while/lstm_cell_50/Sigmoid_2:y:0'while/lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_50/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_50/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_50/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_50/BiasAdd/ReadVariableOp)^while/lstm_cell_50/MatMul/ReadVariableOp+^while/lstm_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_50_biasadd_readvariableop_resource4while_lstm_cell_50_biasadd_readvariableop_resource_0"l
3while_lstm_cell_50_matmul_1_readvariableop_resource5while_lstm_cell_50_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_50_matmul_readvariableop_resource3while_lstm_cell_50_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_50/BiasAdd/ReadVariableOp)while/lstm_cell_50/BiasAdd/ReadVariableOp2T
(while/lstm_cell_50/MatMul/ReadVariableOp(while/lstm_cell_50/MatMul/ReadVariableOp2X
*while/lstm_cell_50/MatMul_1/ReadVariableOp*while/lstm_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

·
(__inference_lstm_50_layer_call_fn_962640
inputs_0
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_50_layer_call_and_return_conditional_losses_961539o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
µ
Ã
while_cond_962038
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_962038___redundant_placeholder04
0while_while_cond_962038___redundant_placeholder14
0while_while_cond_962038___redundant_placeholder24
0while_while_cond_962038___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
d
Î
J__inference_sequential_101_layer_call_and_return_conditional_losses_962608

inputsF
3lstm_50_lstm_cell_50_matmul_readvariableop_resource:	H
5lstm_50_lstm_cell_50_matmul_1_readvariableop_resource:	 C
4lstm_50_lstm_cell_50_biasadd_readvariableop_resource:	:
(dense_202_matmul_readvariableop_resource:  7
)dense_202_biasadd_readvariableop_resource: :
(dense_203_matmul_readvariableop_resource: 7
)dense_203_biasadd_readvariableop_resource:
identity¢ dense_202/BiasAdd/ReadVariableOp¢dense_202/MatMul/ReadVariableOp¢ dense_203/BiasAdd/ReadVariableOp¢dense_203/MatMul/ReadVariableOp¢+lstm_50/lstm_cell_50/BiasAdd/ReadVariableOp¢*lstm_50/lstm_cell_50/MatMul/ReadVariableOp¢,lstm_50/lstm_cell_50/MatMul_1/ReadVariableOp¢lstm_50/whileC
lstm_50/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_50/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_50/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_50/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_50/strided_sliceStridedSlicelstm_50/Shape:output:0$lstm_50/strided_slice/stack:output:0&lstm_50/strided_slice/stack_1:output:0&lstm_50/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_50/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_50/zeros/packedPacklstm_50/strided_slice:output:0lstm_50/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_50/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_50/zerosFilllstm_50/zeros/packed:output:0lstm_50/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
lstm_50/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_50/zeros_1/packedPacklstm_50/strided_slice:output:0!lstm_50/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_50/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_50/zeros_1Filllstm_50/zeros_1/packed:output:0lstm_50/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
lstm_50/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_50/transpose	Transposeinputslstm_50/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
lstm_50/Shape_1Shapelstm_50/transpose:y:0*
T0*
_output_shapes
:g
lstm_50/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_50/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_50/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_50/strided_slice_1StridedSlicelstm_50/Shape_1:output:0&lstm_50/strided_slice_1/stack:output:0(lstm_50/strided_slice_1/stack_1:output:0(lstm_50/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_50/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_50/TensorArrayV2TensorListReserve,lstm_50/TensorArrayV2/element_shape:output:0 lstm_50/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_50/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_50/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_50/transpose:y:0Flstm_50/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_50/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_50/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_50/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_50/strided_slice_2StridedSlicelstm_50/transpose:y:0&lstm_50/strided_slice_2/stack:output:0(lstm_50/strided_slice_2/stack_1:output:0(lstm_50/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstm_50/lstm_cell_50/MatMul/ReadVariableOpReadVariableOp3lstm_50_lstm_cell_50_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0®
lstm_50/lstm_cell_50/MatMulMatMul lstm_50/strided_slice_2:output:02lstm_50/lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,lstm_50/lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp5lstm_50_lstm_cell_50_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0¨
lstm_50/lstm_cell_50/MatMul_1MatMullstm_50/zeros:output:04lstm_50/lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm_50/lstm_cell_50/addAddV2%lstm_50/lstm_cell_50/MatMul:product:0'lstm_50/lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstm_50/lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp4lstm_50_lstm_cell_50_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstm_50/lstm_cell_50/BiasAddBiasAddlstm_50/lstm_cell_50/add:z:03lstm_50/lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstm_50/lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :õ
lstm_50/lstm_cell_50/splitSplit-lstm_50/lstm_cell_50/split/split_dim:output:0%lstm_50/lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split~
lstm_50/lstm_cell_50/SigmoidSigmoid#lstm_50/lstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/lstm_cell_50/Sigmoid_1Sigmoid#lstm_50/lstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/lstm_cell_50/mulMul"lstm_50/lstm_cell_50/Sigmoid_1:y:0lstm_50/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
lstm_50/lstm_cell_50/ReluRelu#lstm_50/lstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/lstm_cell_50/mul_1Mul lstm_50/lstm_cell_50/Sigmoid:y:0'lstm_50/lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/lstm_cell_50/add_1AddV2lstm_50/lstm_cell_50/mul:z:0lstm_50/lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_50/lstm_cell_50/Sigmoid_2Sigmoid#lstm_50/lstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
lstm_50/lstm_cell_50/Relu_1Relulstm_50/lstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¢
lstm_50/lstm_cell_50/mul_2Mul"lstm_50/lstm_cell_50/Sigmoid_2:y:0)lstm_50/lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
%lstm_50/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ð
lstm_50/TensorArrayV2_1TensorListReserve.lstm_50/TensorArrayV2_1/element_shape:output:0 lstm_50/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_50/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_50/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_50/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ò
lstm_50/whileWhile#lstm_50/while/loop_counter:output:0)lstm_50/while/maximum_iterations:output:0lstm_50/time:output:0 lstm_50/TensorArrayV2_1:handle:0lstm_50/zeros:output:0lstm_50/zeros_1:output:0 lstm_50/strided_slice_1:output:0?lstm_50/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_50_lstm_cell_50_matmul_readvariableop_resource5lstm_50_lstm_cell_50_matmul_1_readvariableop_resource4lstm_50_lstm_cell_50_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_50_while_body_962511*%
condR
lstm_50_while_cond_962510*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
8lstm_50/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ú
*lstm_50/TensorArrayV2Stack/TensorListStackTensorListStacklstm_50/while:output:3Alstm_50/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0p
lstm_50/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_50/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_50/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_50/strided_slice_3StridedSlice3lstm_50/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_50/strided_slice_3/stack:output:0(lstm_50/strided_slice_3/stack_1:output:0(lstm_50/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
lstm_50/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_50/transpose_1	Transpose3lstm_50/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_50/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
lstm_50/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_202/MatMul/ReadVariableOpReadVariableOp(dense_202_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_202/MatMulMatMul lstm_50/strided_slice_3:output:0'dense_202/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 dense_202/BiasAdd/ReadVariableOpReadVariableOp)dense_202_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_202/BiasAddBiasAdddense_202/MatMul:product:0(dense_202/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
dense_202/ReluReludense_202/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_203/MatMul/ReadVariableOpReadVariableOp(dense_203_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_203/MatMulMatMuldense_202/Relu:activations:0'dense_203/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_203/BiasAdd/ReadVariableOpReadVariableOp)dense_203_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_203/BiasAddBiasAdddense_203/MatMul:product:0(dense_203/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_203/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp!^dense_202/BiasAdd/ReadVariableOp ^dense_202/MatMul/ReadVariableOp!^dense_203/BiasAdd/ReadVariableOp ^dense_203/MatMul/ReadVariableOp,^lstm_50/lstm_cell_50/BiasAdd/ReadVariableOp+^lstm_50/lstm_cell_50/MatMul/ReadVariableOp-^lstm_50/lstm_cell_50/MatMul_1/ReadVariableOp^lstm_50/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : : : : 2D
 dense_202/BiasAdd/ReadVariableOp dense_202/BiasAdd/ReadVariableOp2B
dense_202/MatMul/ReadVariableOpdense_202/MatMul/ReadVariableOp2D
 dense_203/BiasAdd/ReadVariableOp dense_203/BiasAdd/ReadVariableOp2B
dense_203/MatMul/ReadVariableOpdense_203/MatMul/ReadVariableOp2Z
+lstm_50/lstm_cell_50/BiasAdd/ReadVariableOp+lstm_50/lstm_cell_50/BiasAdd/ReadVariableOp2X
*lstm_50/lstm_cell_50/MatMul/ReadVariableOp*lstm_50/lstm_cell_50/MatMul/ReadVariableOp2\
,lstm_50/lstm_cell_50/MatMul_1/ReadVariableOp,lstm_50/lstm_cell_50/MatMul_1/ReadVariableOp2
lstm_50/whilelstm_50/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±S
³
(sequential_101_lstm_50_while_body_961292J
Fsequential_101_lstm_50_while_sequential_101_lstm_50_while_loop_counterP
Lsequential_101_lstm_50_while_sequential_101_lstm_50_while_maximum_iterations,
(sequential_101_lstm_50_while_placeholder.
*sequential_101_lstm_50_while_placeholder_1.
*sequential_101_lstm_50_while_placeholder_2.
*sequential_101_lstm_50_while_placeholder_3I
Esequential_101_lstm_50_while_sequential_101_lstm_50_strided_slice_1_0
sequential_101_lstm_50_while_tensorarrayv2read_tensorlistgetitem_sequential_101_lstm_50_tensorarrayunstack_tensorlistfromtensor_0]
Jsequential_101_lstm_50_while_lstm_cell_50_matmul_readvariableop_resource_0:	_
Lsequential_101_lstm_50_while_lstm_cell_50_matmul_1_readvariableop_resource_0:	 Z
Ksequential_101_lstm_50_while_lstm_cell_50_biasadd_readvariableop_resource_0:	)
%sequential_101_lstm_50_while_identity+
'sequential_101_lstm_50_while_identity_1+
'sequential_101_lstm_50_while_identity_2+
'sequential_101_lstm_50_while_identity_3+
'sequential_101_lstm_50_while_identity_4+
'sequential_101_lstm_50_while_identity_5G
Csequential_101_lstm_50_while_sequential_101_lstm_50_strided_slice_1
sequential_101_lstm_50_while_tensorarrayv2read_tensorlistgetitem_sequential_101_lstm_50_tensorarrayunstack_tensorlistfromtensor[
Hsequential_101_lstm_50_while_lstm_cell_50_matmul_readvariableop_resource:	]
Jsequential_101_lstm_50_while_lstm_cell_50_matmul_1_readvariableop_resource:	 X
Isequential_101_lstm_50_while_lstm_cell_50_biasadd_readvariableop_resource:	¢@sequential_101/lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOp¢?sequential_101/lstm_50/while/lstm_cell_50/MatMul/ReadVariableOp¢Asequential_101/lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOp
Nsequential_101/lstm_50/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
@sequential_101/lstm_50/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_101_lstm_50_while_tensorarrayv2read_tensorlistgetitem_sequential_101_lstm_50_tensorarrayunstack_tensorlistfromtensor_0(sequential_101_lstm_50_while_placeholderWsequential_101/lstm_50/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Ë
?sequential_101/lstm_50/while/lstm_cell_50/MatMul/ReadVariableOpReadVariableOpJsequential_101_lstm_50_while_lstm_cell_50_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ÿ
0sequential_101/lstm_50/while/lstm_cell_50/MatMulMatMulGsequential_101/lstm_50/while/TensorArrayV2Read/TensorListGetItem:item:0Gsequential_101/lstm_50/while/lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
Asequential_101/lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOpLsequential_101_lstm_50_while_lstm_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0æ
2sequential_101/lstm_50/while/lstm_cell_50/MatMul_1MatMul*sequential_101_lstm_50_while_placeholder_2Isequential_101/lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
-sequential_101/lstm_50/while/lstm_cell_50/addAddV2:sequential_101/lstm_50/while/lstm_cell_50/MatMul:product:0<sequential_101/lstm_50/while/lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
@sequential_101/lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOpKsequential_101_lstm_50_while_lstm_cell_50_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ì
1sequential_101/lstm_50/while/lstm_cell_50/BiasAddBiasAdd1sequential_101/lstm_50/while/lstm_cell_50/add:z:0Hsequential_101/lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
9sequential_101/lstm_50/while/lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :´
/sequential_101/lstm_50/while/lstm_cell_50/splitSplitBsequential_101/lstm_50/while/lstm_cell_50/split/split_dim:output:0:sequential_101/lstm_50/while/lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split¨
1sequential_101/lstm_50/while/lstm_cell_50/SigmoidSigmoid8sequential_101/lstm_50/while/lstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
3sequential_101/lstm_50/while/lstm_cell_50/Sigmoid_1Sigmoid8sequential_101/lstm_50/while/lstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ë
-sequential_101/lstm_50/while/lstm_cell_50/mulMul7sequential_101/lstm_50/while/lstm_cell_50/Sigmoid_1:y:0*sequential_101_lstm_50_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¢
.sequential_101/lstm_50/while/lstm_cell_50/ReluRelu8sequential_101/lstm_50/while/lstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ý
/sequential_101/lstm_50/while/lstm_cell_50/mul_1Mul5sequential_101/lstm_50/while/lstm_cell_50/Sigmoid:y:0<sequential_101/lstm_50/while/lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ò
/sequential_101/lstm_50/while/lstm_cell_50/add_1AddV21sequential_101/lstm_50/while/lstm_cell_50/mul:z:03sequential_101/lstm_50/while/lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
3sequential_101/lstm_50/while/lstm_cell_50/Sigmoid_2Sigmoid8sequential_101/lstm_50/while/lstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
0sequential_101/lstm_50/while/lstm_cell_50/Relu_1Relu3sequential_101/lstm_50/while/lstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ á
/sequential_101/lstm_50/while/lstm_cell_50/mul_2Mul7sequential_101/lstm_50/while/lstm_cell_50/Sigmoid_2:y:0>sequential_101/lstm_50/while/lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
Asequential_101/lstm_50/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*sequential_101_lstm_50_while_placeholder_1(sequential_101_lstm_50_while_placeholder3sequential_101/lstm_50/while/lstm_cell_50/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒd
"sequential_101/lstm_50/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¡
 sequential_101/lstm_50/while/addAddV2(sequential_101_lstm_50_while_placeholder+sequential_101/lstm_50/while/add/y:output:0*
T0*
_output_shapes
: f
$sequential_101/lstm_50/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ã
"sequential_101/lstm_50/while/add_1AddV2Fsequential_101_lstm_50_while_sequential_101_lstm_50_while_loop_counter-sequential_101/lstm_50/while/add_1/y:output:0*
T0*
_output_shapes
: 
%sequential_101/lstm_50/while/IdentityIdentity&sequential_101/lstm_50/while/add_1:z:0"^sequential_101/lstm_50/while/NoOp*
T0*
_output_shapes
: Æ
'sequential_101/lstm_50/while/Identity_1IdentityLsequential_101_lstm_50_while_sequential_101_lstm_50_while_maximum_iterations"^sequential_101/lstm_50/while/NoOp*
T0*
_output_shapes
: 
'sequential_101/lstm_50/while/Identity_2Identity$sequential_101/lstm_50/while/add:z:0"^sequential_101/lstm_50/while/NoOp*
T0*
_output_shapes
: Þ
'sequential_101/lstm_50/while/Identity_3IdentityQsequential_101/lstm_50/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^sequential_101/lstm_50/while/NoOp*
T0*
_output_shapes
: :éèÒ¾
'sequential_101/lstm_50/while/Identity_4Identity3sequential_101/lstm_50/while/lstm_cell_50/mul_2:z:0"^sequential_101/lstm_50/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¾
'sequential_101/lstm_50/while/Identity_5Identity3sequential_101/lstm_50/while/lstm_cell_50/add_1:z:0"^sequential_101/lstm_50/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
!sequential_101/lstm_50/while/NoOpNoOpA^sequential_101/lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOp@^sequential_101/lstm_50/while/lstm_cell_50/MatMul/ReadVariableOpB^sequential_101/lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "W
%sequential_101_lstm_50_while_identity.sequential_101/lstm_50/while/Identity:output:0"[
'sequential_101_lstm_50_while_identity_10sequential_101/lstm_50/while/Identity_1:output:0"[
'sequential_101_lstm_50_while_identity_20sequential_101/lstm_50/while/Identity_2:output:0"[
'sequential_101_lstm_50_while_identity_30sequential_101/lstm_50/while/Identity_3:output:0"[
'sequential_101_lstm_50_while_identity_40sequential_101/lstm_50/while/Identity_4:output:0"[
'sequential_101_lstm_50_while_identity_50sequential_101/lstm_50/while/Identity_5:output:0"
Isequential_101_lstm_50_while_lstm_cell_50_biasadd_readvariableop_resourceKsequential_101_lstm_50_while_lstm_cell_50_biasadd_readvariableop_resource_0"
Jsequential_101_lstm_50_while_lstm_cell_50_matmul_1_readvariableop_resourceLsequential_101_lstm_50_while_lstm_cell_50_matmul_1_readvariableop_resource_0"
Hsequential_101_lstm_50_while_lstm_cell_50_matmul_readvariableop_resourceJsequential_101_lstm_50_while_lstm_cell_50_matmul_readvariableop_resource_0"
Csequential_101_lstm_50_while_sequential_101_lstm_50_strided_slice_1Esequential_101_lstm_50_while_sequential_101_lstm_50_strided_slice_1_0"
sequential_101_lstm_50_while_tensorarrayv2read_tensorlistgetitem_sequential_101_lstm_50_tensorarrayunstack_tensorlistfromtensorsequential_101_lstm_50_while_tensorarrayv2read_tensorlistgetitem_sequential_101_lstm_50_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2
@sequential_101/lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOp@sequential_101/lstm_50/while/lstm_cell_50/BiasAdd/ReadVariableOp2
?sequential_101/lstm_50/while/lstm_cell_50/MatMul/ReadVariableOp?sequential_101/lstm_50/while/lstm_cell_50/MatMul/ReadVariableOp2
Asequential_101/lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOpAsequential_101/lstm_50/while/lstm_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ý

H__inference_lstm_cell_50_layer_call_and_return_conditional_losses_963350

inputs
states_0
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
8
Ð
while_body_962039
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_50_matmul_readvariableop_resource_0:	H
5while_lstm_cell_50_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_50_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_50_matmul_readvariableop_resource:	F
3while_lstm_cell_50_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_50_biasadd_readvariableop_resource:	¢)while/lstm_cell_50/BiasAdd/ReadVariableOp¢(while/lstm_cell_50/MatMul/ReadVariableOp¢*while/lstm_cell_50/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_50/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_50_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_50/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_50/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_50/addAddV2#while/lstm_cell_50/MatMul:product:0%while/lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_50_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_50/BiasAddBiasAddwhile/lstm_cell_50/add:z:01while/lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_50/splitSplit+while/lstm_cell_50/split/split_dim:output:0#while/lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_50/SigmoidSigmoid!while/lstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_50/Sigmoid_1Sigmoid!while/lstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mulMul while/lstm_cell_50/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_50/ReluRelu!while/lstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mul_1Mulwhile/lstm_cell_50/Sigmoid:y:0%while/lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/add_1AddV2while/lstm_cell_50/mul:z:0while/lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_50/Sigmoid_2Sigmoid!while/lstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_50/Relu_1Reluwhile/lstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mul_2Mul while/lstm_cell_50/Sigmoid_2:y:0'while/lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_50/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_50/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_50/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_50/BiasAdd/ReadVariableOp)^while/lstm_cell_50/MatMul/ReadVariableOp+^while/lstm_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_50_biasadd_readvariableop_resource4while_lstm_cell_50_biasadd_readvariableop_resource_0"l
3while_lstm_cell_50_matmul_1_readvariableop_resource5while_lstm_cell_50_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_50_matmul_readvariableop_resource3while_lstm_cell_50_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_50/BiasAdd/ReadVariableOp)while/lstm_cell_50/BiasAdd/ReadVariableOp2T
(while/lstm_cell_50/MatMul/ReadVariableOp(while/lstm_cell_50/MatMul/ReadVariableOp2X
*while/lstm_cell_50/MatMul_1/ReadVariableOp*while/lstm_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ÝJ

C__inference_lstm_50_layer_call_and_return_conditional_losses_962959
inputs_0>
+lstm_cell_50_matmul_readvariableop_resource:	@
-lstm_cell_50_matmul_1_readvariableop_resource:	 ;
,lstm_cell_50_biasadd_readvariableop_resource:	
identity¢#lstm_cell_50/BiasAdd/ReadVariableOp¢"lstm_cell_50/MatMul/ReadVariableOp¢$lstm_cell_50/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_50/MatMul/ReadVariableOpReadVariableOp+lstm_cell_50_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_50/MatMulMatMulstrided_slice_2:output:0*lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_50_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_50/MatMul_1MatMulzeros:output:0,lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_50/addAddV2lstm_cell_50/MatMul:product:0lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_50_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_50/BiasAddBiasAddlstm_cell_50/add:z:0+lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_50/splitSplit%lstm_cell_50/split/split_dim:output:0lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_50/SigmoidSigmoidlstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_50/Sigmoid_1Sigmoidlstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_50/mulMullstm_cell_50/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_50/ReluRelulstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_50/mul_1Mullstm_cell_50/Sigmoid:y:0lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_50/add_1AddV2lstm_cell_50/mul:z:0lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_50/Sigmoid_2Sigmoidlstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_50/Relu_1Relulstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_50/mul_2Mullstm_cell_50/Sigmoid_2:y:0!lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_50_matmul_readvariableop_resource-lstm_cell_50_matmul_1_readvariableop_resource,lstm_cell_50_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_962875*
condR
while_cond_962874*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_50/BiasAdd/ReadVariableOp#^lstm_cell_50/MatMul/ReadVariableOp%^lstm_cell_50/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_50/BiasAdd/ReadVariableOp#lstm_cell_50/BiasAdd/ReadVariableOp2H
"lstm_cell_50/MatMul/ReadVariableOp"lstm_cell_50/MatMul/ReadVariableOp2L
$lstm_cell_50/MatMul_1/ReadVariableOp$lstm_cell_50/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ç
Ç
J__inference_sequential_101_layer_call_and_return_conditional_losses_962231
lstm_50_input!
lstm_50_962213:	!
lstm_50_962215:	 
lstm_50_962217:	"
dense_202_962220:  
dense_202_962222: "
dense_203_962225: 
dense_203_962227:
identity¢!dense_202/StatefulPartitionedCall¢!dense_203/StatefulPartitionedCall¢lstm_50/StatefulPartitionedCall
lstm_50/StatefulPartitionedCallStatefulPartitionedCalllstm_50_inputlstm_50_962213lstm_50_962215lstm_50_962217*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_50_layer_call_and_return_conditional_losses_961888
!dense_202/StatefulPartitionedCallStatefulPartitionedCall(lstm_50/StatefulPartitionedCall:output:0dense_202_962220dense_202_962222*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_202_layer_call_and_return_conditional_losses_961907
!dense_203/StatefulPartitionedCallStatefulPartitionedCall*dense_202/StatefulPartitionedCall:output:0dense_203_962225dense_203_962227*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_203_layer_call_and_return_conditional_losses_961923y
IdentityIdentity*dense_203/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
NoOpNoOp"^dense_202/StatefulPartitionedCall"^dense_203/StatefulPartitionedCall ^lstm_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : : : : 2F
!dense_202/StatefulPartitionedCall!dense_202/StatefulPartitionedCall2F
!dense_203/StatefulPartitionedCall!dense_203/StatefulPartitionedCall2B
lstm_50/StatefulPartitionedCalllstm_50/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_50_input
	
¤
$__inference_signature_wrapper_962629
lstm_50_input
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCalllstm_50_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_961389o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_50_input
ç
Ç
J__inference_sequential_101_layer_call_and_return_conditional_losses_962252
lstm_50_input!
lstm_50_962234:	!
lstm_50_962236:	 
lstm_50_962238:	"
dense_202_962241:  
dense_202_962243: "
dense_203_962246: 
dense_203_962248:
identity¢!dense_202/StatefulPartitionedCall¢!dense_203/StatefulPartitionedCall¢lstm_50/StatefulPartitionedCall
lstm_50/StatefulPartitionedCallStatefulPartitionedCalllstm_50_inputlstm_50_962234lstm_50_962236lstm_50_962238*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_50_layer_call_and_return_conditional_losses_962123
!dense_202/StatefulPartitionedCallStatefulPartitionedCall(lstm_50/StatefulPartitionedCall:output:0dense_202_962241dense_202_962243*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_202_layer_call_and_return_conditional_losses_961907
!dense_203/StatefulPartitionedCallStatefulPartitionedCall*dense_202/StatefulPartitionedCall:output:0dense_203_962246dense_203_962248*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_203_layer_call_and_return_conditional_losses_961923y
IdentityIdentity*dense_203/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
NoOpNoOp"^dense_202/StatefulPartitionedCall"^dense_203/StatefulPartitionedCall ^lstm_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : : : : 2F
!dense_202/StatefulPartitionedCall!dense_202/StatefulPartitionedCall2F
!dense_203/StatefulPartitionedCall!dense_203/StatefulPartitionedCall2B
lstm_50/StatefulPartitionedCalllstm_50/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_50_input
È	
ö
E__inference_dense_203_layer_call_and_return_conditional_losses_961923

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
8

C__inference_lstm_50_layer_call_and_return_conditional_losses_961539

inputs&
lstm_cell_50_961457:	&
lstm_cell_50_961459:	 "
lstm_cell_50_961461:	
identity¢$lstm_cell_50/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskõ
$lstm_cell_50/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_50_961457lstm_cell_50_961459lstm_cell_50_961461*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_50_layer_call_and_return_conditional_losses_961456n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_50_961457lstm_cell_50_961459lstm_cell_50_961461*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_961470*
condR
while_cond_961469*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
NoOpNoOp%^lstm_cell_50/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_50/StatefulPartitionedCall$lstm_cell_50/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â	
¯
/__inference_sequential_101_layer_call_fn_961947
lstm_50_input
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCalllstm_50_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_101_layer_call_and_return_conditional_losses_961930o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_50_input


ã
lstm_50_while_cond_962354,
(lstm_50_while_lstm_50_while_loop_counter2
.lstm_50_while_lstm_50_while_maximum_iterations
lstm_50_while_placeholder
lstm_50_while_placeholder_1
lstm_50_while_placeholder_2
lstm_50_while_placeholder_3.
*lstm_50_while_less_lstm_50_strided_slice_1D
@lstm_50_while_lstm_50_while_cond_962354___redundant_placeholder0D
@lstm_50_while_lstm_50_while_cond_962354___redundant_placeholder1D
@lstm_50_while_lstm_50_while_cond_962354___redundant_placeholder2D
@lstm_50_while_lstm_50_while_cond_962354___redundant_placeholder3
lstm_50_while_identity

lstm_50/while/LessLesslstm_50_while_placeholder*lstm_50_while_less_lstm_50_strided_slice_1*
T0*
_output_shapes
: [
lstm_50/while/IdentityIdentitylstm_50/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_50_while_identitylstm_50/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
µ
Ã
while_cond_961469
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_961469___redundant_placeholder04
0while_while_cond_961469___redundant_placeholder14
0while_while_cond_961469___redundant_placeholder24
0while_while_cond_961469___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ù"
ã
while_body_961661
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_50_961685_0:	.
while_lstm_cell_50_961687_0:	 *
while_lstm_cell_50_961689_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_50_961685:	,
while_lstm_cell_50_961687:	 (
while_lstm_cell_50_961689:	¢*while/lstm_cell_50/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0³
*while/lstm_cell_50/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_50_961685_0while_lstm_cell_50_961687_0while_lstm_cell_50_961689_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_50_layer_call_and_return_conditional_losses_961602Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_50/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity3while/lstm_cell_50/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/Identity_5Identity3while/lstm_cell_50/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y

while/NoOpNoOp+^while/lstm_cell_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_50_961685while_lstm_cell_50_961685_0"8
while_lstm_cell_50_961687while_lstm_cell_50_961687_0"8
while_lstm_cell_50_961689while_lstm_cell_50_961689_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_50/StatefulPartitionedCall*while/lstm_cell_50/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
8
Ð
while_body_962732
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_50_matmul_readvariableop_resource_0:	H
5while_lstm_cell_50_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_50_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_50_matmul_readvariableop_resource:	F
3while_lstm_cell_50_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_50_biasadd_readvariableop_resource:	¢)while/lstm_cell_50/BiasAdd/ReadVariableOp¢(while/lstm_cell_50/MatMul/ReadVariableOp¢*while/lstm_cell_50/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_50/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_50_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_50/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_50/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_50/addAddV2#while/lstm_cell_50/MatMul:product:0%while/lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_50_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_50/BiasAddBiasAddwhile/lstm_cell_50/add:z:01while/lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_50/splitSplit+while/lstm_cell_50/split/split_dim:output:0#while/lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_50/SigmoidSigmoid!while/lstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_50/Sigmoid_1Sigmoid!while/lstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mulMul while/lstm_cell_50/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_50/ReluRelu!while/lstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mul_1Mulwhile/lstm_cell_50/Sigmoid:y:0%while/lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/add_1AddV2while/lstm_cell_50/mul:z:0while/lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_50/Sigmoid_2Sigmoid!while/lstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_50/Relu_1Reluwhile/lstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mul_2Mul while/lstm_cell_50/Sigmoid_2:y:0'while/lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_50/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_50/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_50/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_50/BiasAdd/ReadVariableOp)^while/lstm_cell_50/MatMul/ReadVariableOp+^while/lstm_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_50_biasadd_readvariableop_resource4while_lstm_cell_50_biasadd_readvariableop_resource_0"l
3while_lstm_cell_50_matmul_1_readvariableop_resource5while_lstm_cell_50_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_50_matmul_readvariableop_resource3while_lstm_cell_50_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_50/BiasAdd/ReadVariableOp)while/lstm_cell_50/BiasAdd/ReadVariableOp2T
(while/lstm_cell_50/MatMul/ReadVariableOp(while/lstm_cell_50/MatMul/ReadVariableOp2X
*while/lstm_cell_50/MatMul_1/ReadVariableOp*while/lstm_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Õ

H__inference_lstm_cell_50_layer_call_and_return_conditional_losses_961602

inputs

states
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
8

C__inference_lstm_50_layer_call_and_return_conditional_losses_961730

inputs&
lstm_cell_50_961648:	&
lstm_cell_50_961650:	 "
lstm_cell_50_961652:	
identity¢$lstm_cell_50/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskõ
$lstm_cell_50/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_50_961648lstm_cell_50_961650lstm_cell_50_961652*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_50_layer_call_and_return_conditional_losses_961602n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_50_961648lstm_cell_50_961650lstm_cell_50_961652*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_961661*
condR
while_cond_961660*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
NoOpNoOp%^lstm_cell_50/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_50/StatefulPartitionedCall$lstm_cell_50/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Úr

"__inference__traced_restore_963583
file_prefix3
!assignvariableop_dense_202_kernel:  /
!assignvariableop_1_dense_202_bias: 5
#assignvariableop_2_dense_203_kernel: /
!assignvariableop_3_dense_203_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: A
.assignvariableop_9_lstm_50_lstm_cell_50_kernel:	L
9assignvariableop_10_lstm_50_lstm_cell_50_recurrent_kernel:	 <
-assignvariableop_11_lstm_50_lstm_cell_50_bias:	#
assignvariableop_12_total: #
assignvariableop_13_count: =
+assignvariableop_14_adam_dense_202_kernel_m:  7
)assignvariableop_15_adam_dense_202_bias_m: =
+assignvariableop_16_adam_dense_203_kernel_m: 7
)assignvariableop_17_adam_dense_203_bias_m:I
6assignvariableop_18_adam_lstm_50_lstm_cell_50_kernel_m:	S
@assignvariableop_19_adam_lstm_50_lstm_cell_50_recurrent_kernel_m:	 C
4assignvariableop_20_adam_lstm_50_lstm_cell_50_bias_m:	=
+assignvariableop_21_adam_dense_202_kernel_v:  7
)assignvariableop_22_adam_dense_202_bias_v: =
+assignvariableop_23_adam_dense_203_kernel_v: 7
)assignvariableop_24_adam_dense_203_bias_v:I
6assignvariableop_25_adam_lstm_50_lstm_cell_50_kernel_v:	S
@assignvariableop_26_adam_lstm_50_lstm_cell_50_recurrent_kernel_v:	 C
4assignvariableop_27_adam_lstm_50_lstm_cell_50_bias_v:	
identity_29¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9â
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueþBûB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHª
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B °
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_dense_202_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_202_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_203_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_203_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp.assignvariableop_9_lstm_50_lstm_cell_50_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_10AssignVariableOp9assignvariableop_10_lstm_50_lstm_cell_50_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp-assignvariableop_11_lstm_50_lstm_cell_50_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp+assignvariableop_14_adam_dense_202_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_202_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_dense_203_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_203_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_lstm_50_lstm_cell_50_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_19AssignVariableOp@assignvariableop_19_adam_lstm_50_lstm_cell_50_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_lstm_50_lstm_cell_50_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_202_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_202_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_203_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_203_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_50_lstm_cell_50_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_lstm_50_lstm_cell_50_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_50_lstm_cell_50_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ·
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: ¤
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ë
ö
-__inference_lstm_cell_50_layer_call_fn_963301

inputs
states_0
states_1
unknown:	
	unknown_0:	 
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_50_layer_call_and_return_conditional_losses_961456o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
­	
¨
/__inference_sequential_101_layer_call_fn_962296

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_101_layer_call_and_return_conditional_losses_962174o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨J

C__inference_lstm_50_layer_call_and_return_conditional_losses_962123

inputs>
+lstm_cell_50_matmul_readvariableop_resource:	@
-lstm_cell_50_matmul_1_readvariableop_resource:	 ;
,lstm_cell_50_biasadd_readvariableop_resource:	
identity¢#lstm_cell_50/BiasAdd/ReadVariableOp¢"lstm_cell_50/MatMul/ReadVariableOp¢$lstm_cell_50/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_50/MatMul/ReadVariableOpReadVariableOp+lstm_cell_50_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_50/MatMulMatMulstrided_slice_2:output:0*lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_50_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_50/MatMul_1MatMulzeros:output:0,lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_50/addAddV2lstm_cell_50/MatMul:product:0lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_50_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_50/BiasAddBiasAddlstm_cell_50/add:z:0+lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_50/splitSplit%lstm_cell_50/split/split_dim:output:0lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_50/SigmoidSigmoidlstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_50/Sigmoid_1Sigmoidlstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_50/mulMullstm_cell_50/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_50/ReluRelulstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_50/mul_1Mullstm_cell_50/Sigmoid:y:0lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_50/add_1AddV2lstm_cell_50/mul:z:0lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_50/Sigmoid_2Sigmoidlstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_50/Relu_1Relulstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_50/mul_2Mullstm_cell_50/Sigmoid_2:y:0!lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_50_matmul_readvariableop_resource-lstm_cell_50_matmul_1_readvariableop_resource,lstm_cell_50_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_962039*
condR
while_cond_962038*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_50/BiasAdd/ReadVariableOp#^lstm_cell_50/MatMul/ReadVariableOp%^lstm_cell_50/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_50/BiasAdd/ReadVariableOp#lstm_cell_50/BiasAdd/ReadVariableOp2H
"lstm_cell_50/MatMul/ReadVariableOp"lstm_cell_50/MatMul/ReadVariableOp2L
$lstm_cell_50/MatMul_1/ReadVariableOp$lstm_cell_50/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ã
lstm_50_while_cond_962510,
(lstm_50_while_lstm_50_while_loop_counter2
.lstm_50_while_lstm_50_while_maximum_iterations
lstm_50_while_placeholder
lstm_50_while_placeholder_1
lstm_50_while_placeholder_2
lstm_50_while_placeholder_3.
*lstm_50_while_less_lstm_50_strided_slice_1D
@lstm_50_while_lstm_50_while_cond_962510___redundant_placeholder0D
@lstm_50_while_lstm_50_while_cond_962510___redundant_placeholder1D
@lstm_50_while_lstm_50_while_cond_962510___redundant_placeholder2D
@lstm_50_while_lstm_50_while_cond_962510___redundant_placeholder3
lstm_50_while_identity

lstm_50/while/LessLesslstm_50_while_placeholder*lstm_50_while_less_lstm_50_strided_slice_1*
T0*
_output_shapes
: [
lstm_50/while/IdentityIdentitylstm_50/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_50_while_identitylstm_50/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Ä

*__inference_dense_203_layer_call_fn_963274

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_203_layer_call_and_return_conditional_losses_961923o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
í?
î
__inference__traced_save_963489
file_prefix/
+savev2_dense_202_kernel_read_readvariableop-
)savev2_dense_202_bias_read_readvariableop/
+savev2_dense_203_kernel_read_readvariableop-
)savev2_dense_203_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_50_lstm_cell_50_kernel_read_readvariableopD
@savev2_lstm_50_lstm_cell_50_recurrent_kernel_read_readvariableop8
4savev2_lstm_50_lstm_cell_50_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_202_kernel_m_read_readvariableop4
0savev2_adam_dense_202_bias_m_read_readvariableop6
2savev2_adam_dense_203_kernel_m_read_readvariableop4
0savev2_adam_dense_203_bias_m_read_readvariableopA
=savev2_adam_lstm_50_lstm_cell_50_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_50_lstm_cell_50_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_50_lstm_cell_50_bias_m_read_readvariableop6
2savev2_adam_dense_202_kernel_v_read_readvariableop4
0savev2_adam_dense_202_bias_v_read_readvariableop6
2savev2_adam_dense_203_kernel_v_read_readvariableop4
0savev2_adam_dense_203_bias_v_read_readvariableopA
=savev2_adam_lstm_50_lstm_cell_50_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_50_lstm_cell_50_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_50_lstm_cell_50_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ß
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueþBûB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH§
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B Þ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_202_kernel_read_readvariableop)savev2_dense_202_bias_read_readvariableop+savev2_dense_203_kernel_read_readvariableop)savev2_dense_203_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_50_lstm_cell_50_kernel_read_readvariableop@savev2_lstm_50_lstm_cell_50_recurrent_kernel_read_readvariableop4savev2_lstm_50_lstm_cell_50_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_202_kernel_m_read_readvariableop0savev2_adam_dense_202_bias_m_read_readvariableop2savev2_adam_dense_203_kernel_m_read_readvariableop0savev2_adam_dense_203_bias_m_read_readvariableop=savev2_adam_lstm_50_lstm_cell_50_kernel_m_read_readvariableopGsavev2_adam_lstm_50_lstm_cell_50_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_50_lstm_cell_50_bias_m_read_readvariableop2savev2_adam_dense_202_kernel_v_read_readvariableop0savev2_adam_dense_202_bias_v_read_readvariableop2savev2_adam_dense_203_kernel_v_read_readvariableop0savev2_adam_dense_203_bias_v_read_readvariableop=savev2_adam_lstm_50_lstm_cell_50_kernel_v_read_readvariableopGsavev2_adam_lstm_50_lstm_cell_50_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_50_lstm_cell_50_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Þ
_input_shapesÌ
É: :  : : :: : : : : :	:	 :: : :  : : ::	:	 ::  : : ::	:	 :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
::$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
::

_output_shapes
: 
ª{

!__inference__wrapped_model_961389
lstm_50_inputU
Bsequential_101_lstm_50_lstm_cell_50_matmul_readvariableop_resource:	W
Dsequential_101_lstm_50_lstm_cell_50_matmul_1_readvariableop_resource:	 R
Csequential_101_lstm_50_lstm_cell_50_biasadd_readvariableop_resource:	I
7sequential_101_dense_202_matmul_readvariableop_resource:  F
8sequential_101_dense_202_biasadd_readvariableop_resource: I
7sequential_101_dense_203_matmul_readvariableop_resource: F
8sequential_101_dense_203_biasadd_readvariableop_resource:
identity¢/sequential_101/dense_202/BiasAdd/ReadVariableOp¢.sequential_101/dense_202/MatMul/ReadVariableOp¢/sequential_101/dense_203/BiasAdd/ReadVariableOp¢.sequential_101/dense_203/MatMul/ReadVariableOp¢:sequential_101/lstm_50/lstm_cell_50/BiasAdd/ReadVariableOp¢9sequential_101/lstm_50/lstm_cell_50/MatMul/ReadVariableOp¢;sequential_101/lstm_50/lstm_cell_50/MatMul_1/ReadVariableOp¢sequential_101/lstm_50/whileY
sequential_101/lstm_50/ShapeShapelstm_50_input*
T0*
_output_shapes
:t
*sequential_101/lstm_50/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_101/lstm_50/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_101/lstm_50/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$sequential_101/lstm_50/strided_sliceStridedSlice%sequential_101/lstm_50/Shape:output:03sequential_101/lstm_50/strided_slice/stack:output:05sequential_101/lstm_50/strided_slice/stack_1:output:05sequential_101/lstm_50/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%sequential_101/lstm_50/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ¸
#sequential_101/lstm_50/zeros/packedPack-sequential_101/lstm_50/strided_slice:output:0.sequential_101/lstm_50/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_101/lstm_50/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
sequential_101/lstm_50/zerosFill,sequential_101/lstm_50/zeros/packed:output:0+sequential_101/lstm_50/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
'sequential_101/lstm_50/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ¼
%sequential_101/lstm_50/zeros_1/packedPack-sequential_101/lstm_50/strided_slice:output:00sequential_101/lstm_50/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:i
$sequential_101/lstm_50/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ·
sequential_101/lstm_50/zeros_1Fill.sequential_101/lstm_50/zeros_1/packed:output:0-sequential_101/lstm_50/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
%sequential_101/lstm_50/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¢
 sequential_101/lstm_50/transpose	Transposelstm_50_input.sequential_101/lstm_50/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
sequential_101/lstm_50/Shape_1Shape$sequential_101/lstm_50/transpose:y:0*
T0*
_output_shapes
:v
,sequential_101/lstm_50/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_101/lstm_50/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_101/lstm_50/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&sequential_101/lstm_50/strided_slice_1StridedSlice'sequential_101/lstm_50/Shape_1:output:05sequential_101/lstm_50/strided_slice_1/stack:output:07sequential_101/lstm_50/strided_slice_1/stack_1:output:07sequential_101/lstm_50/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2sequential_101/lstm_50/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿù
$sequential_101/lstm_50/TensorArrayV2TensorListReserve;sequential_101/lstm_50/TensorArrayV2/element_shape:output:0/sequential_101/lstm_50/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Lsequential_101/lstm_50/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¥
>sequential_101/lstm_50/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$sequential_101/lstm_50/transpose:y:0Usequential_101/lstm_50/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒv
,sequential_101/lstm_50/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_101/lstm_50/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_101/lstm_50/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
&sequential_101/lstm_50/strided_slice_2StridedSlice$sequential_101/lstm_50/transpose:y:05sequential_101/lstm_50/strided_slice_2/stack:output:07sequential_101/lstm_50/strided_slice_2/stack_1:output:07sequential_101/lstm_50/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask½
9sequential_101/lstm_50/lstm_cell_50/MatMul/ReadVariableOpReadVariableOpBsequential_101_lstm_50_lstm_cell_50_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Û
*sequential_101/lstm_50/lstm_cell_50/MatMulMatMul/sequential_101/lstm_50/strided_slice_2:output:0Asequential_101/lstm_50/lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
;sequential_101/lstm_50/lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOpDsequential_101_lstm_50_lstm_cell_50_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0Õ
,sequential_101/lstm_50/lstm_cell_50/MatMul_1MatMul%sequential_101/lstm_50/zeros:output:0Csequential_101/lstm_50/lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
'sequential_101/lstm_50/lstm_cell_50/addAddV24sequential_101/lstm_50/lstm_cell_50/MatMul:product:06sequential_101/lstm_50/lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
:sequential_101/lstm_50/lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOpCsequential_101_lstm_50_lstm_cell_50_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ú
+sequential_101/lstm_50/lstm_cell_50/BiasAddBiasAdd+sequential_101/lstm_50/lstm_cell_50/add:z:0Bsequential_101/lstm_50/lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
3sequential_101/lstm_50/lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¢
)sequential_101/lstm_50/lstm_cell_50/splitSplit<sequential_101/lstm_50/lstm_cell_50/split/split_dim:output:04sequential_101/lstm_50/lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
+sequential_101/lstm_50/lstm_cell_50/SigmoidSigmoid2sequential_101/lstm_50/lstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_101/lstm_50/lstm_cell_50/Sigmoid_1Sigmoid2sequential_101/lstm_50/lstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¼
'sequential_101/lstm_50/lstm_cell_50/mulMul1sequential_101/lstm_50/lstm_cell_50/Sigmoid_1:y:0'sequential_101/lstm_50/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(sequential_101/lstm_50/lstm_cell_50/ReluRelu2sequential_101/lstm_50/lstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ë
)sequential_101/lstm_50/lstm_cell_50/mul_1Mul/sequential_101/lstm_50/lstm_cell_50/Sigmoid:y:06sequential_101/lstm_50/lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
)sequential_101/lstm_50/lstm_cell_50/add_1AddV2+sequential_101/lstm_50/lstm_cell_50/mul:z:0-sequential_101/lstm_50/lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_101/lstm_50/lstm_cell_50/Sigmoid_2Sigmoid2sequential_101/lstm_50/lstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
*sequential_101/lstm_50/lstm_cell_50/Relu_1Relu-sequential_101/lstm_50/lstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ï
)sequential_101/lstm_50/lstm_cell_50/mul_2Mul1sequential_101/lstm_50/lstm_cell_50/Sigmoid_2:y:08sequential_101/lstm_50/lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
4sequential_101/lstm_50/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ý
&sequential_101/lstm_50/TensorArrayV2_1TensorListReserve=sequential_101/lstm_50/TensorArrayV2_1/element_shape:output:0/sequential_101/lstm_50/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ]
sequential_101/lstm_50/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/sequential_101/lstm_50/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿk
)sequential_101/lstm_50/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ä
sequential_101/lstm_50/whileWhile2sequential_101/lstm_50/while/loop_counter:output:08sequential_101/lstm_50/while/maximum_iterations:output:0$sequential_101/lstm_50/time:output:0/sequential_101/lstm_50/TensorArrayV2_1:handle:0%sequential_101/lstm_50/zeros:output:0'sequential_101/lstm_50/zeros_1:output:0/sequential_101/lstm_50/strided_slice_1:output:0Nsequential_101/lstm_50/TensorArrayUnstack/TensorListFromTensor:output_handle:0Bsequential_101_lstm_50_lstm_cell_50_matmul_readvariableop_resourceDsequential_101_lstm_50_lstm_cell_50_matmul_1_readvariableop_resourceCsequential_101_lstm_50_lstm_cell_50_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_101_lstm_50_while_body_961292*4
cond,R*
(sequential_101_lstm_50_while_cond_961291*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
Gsequential_101/lstm_50/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
9sequential_101/lstm_50/TensorArrayV2Stack/TensorListStackTensorListStack%sequential_101/lstm_50/while:output:3Psequential_101/lstm_50/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
,sequential_101/lstm_50/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.sequential_101/lstm_50/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.sequential_101/lstm_50/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ú
&sequential_101/lstm_50/strided_slice_3StridedSliceBsequential_101/lstm_50/TensorArrayV2Stack/TensorListStack:tensor:05sequential_101/lstm_50/strided_slice_3/stack:output:07sequential_101/lstm_50/strided_slice_3/stack_1:output:07sequential_101/lstm_50/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask|
'sequential_101/lstm_50/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Û
"sequential_101/lstm_50/transpose_1	TransposeBsequential_101/lstm_50/TensorArrayV2Stack/TensorListStack:tensor:00sequential_101/lstm_50/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
sequential_101/lstm_50/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ¦
.sequential_101/dense_202/MatMul/ReadVariableOpReadVariableOp7sequential_101_dense_202_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ä
sequential_101/dense_202/MatMulMatMul/sequential_101/lstm_50/strided_slice_3:output:06sequential_101/dense_202/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
/sequential_101/dense_202/BiasAdd/ReadVariableOpReadVariableOp8sequential_101_dense_202_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Á
 sequential_101/dense_202/BiasAddBiasAdd)sequential_101/dense_202/MatMul:product:07sequential_101/dense_202/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
sequential_101/dense_202/ReluRelu)sequential_101/dense_202/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
.sequential_101/dense_203/MatMul/ReadVariableOpReadVariableOp7sequential_101_dense_203_matmul_readvariableop_resource*
_output_shapes

: *
dtype0À
sequential_101/dense_203/MatMulMatMul+sequential_101/dense_202/Relu:activations:06sequential_101/dense_203/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/sequential_101/dense_203/BiasAdd/ReadVariableOpReadVariableOp8sequential_101_dense_203_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 sequential_101/dense_203/BiasAddBiasAdd)sequential_101/dense_203/MatMul:product:07sequential_101/dense_203/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
IdentityIdentity)sequential_101/dense_203/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ
NoOpNoOp0^sequential_101/dense_202/BiasAdd/ReadVariableOp/^sequential_101/dense_202/MatMul/ReadVariableOp0^sequential_101/dense_203/BiasAdd/ReadVariableOp/^sequential_101/dense_203/MatMul/ReadVariableOp;^sequential_101/lstm_50/lstm_cell_50/BiasAdd/ReadVariableOp:^sequential_101/lstm_50/lstm_cell_50/MatMul/ReadVariableOp<^sequential_101/lstm_50/lstm_cell_50/MatMul_1/ReadVariableOp^sequential_101/lstm_50/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : : : : 2b
/sequential_101/dense_202/BiasAdd/ReadVariableOp/sequential_101/dense_202/BiasAdd/ReadVariableOp2`
.sequential_101/dense_202/MatMul/ReadVariableOp.sequential_101/dense_202/MatMul/ReadVariableOp2b
/sequential_101/dense_203/BiasAdd/ReadVariableOp/sequential_101/dense_203/BiasAdd/ReadVariableOp2`
.sequential_101/dense_203/MatMul/ReadVariableOp.sequential_101/dense_203/MatMul/ReadVariableOp2x
:sequential_101/lstm_50/lstm_cell_50/BiasAdd/ReadVariableOp:sequential_101/lstm_50/lstm_cell_50/BiasAdd/ReadVariableOp2v
9sequential_101/lstm_50/lstm_cell_50/MatMul/ReadVariableOp9sequential_101/lstm_50/lstm_cell_50/MatMul/ReadVariableOp2z
;sequential_101/lstm_50/lstm_cell_50/MatMul_1/ReadVariableOp;sequential_101/lstm_50/lstm_cell_50/MatMul_1/ReadVariableOp2<
sequential_101/lstm_50/whilesequential_101/lstm_50/while:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_50_input

·
(__inference_lstm_50_layer_call_fn_962651
inputs_0
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_50_layer_call_and_return_conditional_losses_961730o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¨J

C__inference_lstm_50_layer_call_and_return_conditional_losses_963245

inputs>
+lstm_cell_50_matmul_readvariableop_resource:	@
-lstm_cell_50_matmul_1_readvariableop_resource:	 ;
,lstm_cell_50_biasadd_readvariableop_resource:	
identity¢#lstm_cell_50/BiasAdd/ReadVariableOp¢"lstm_cell_50/MatMul/ReadVariableOp¢$lstm_cell_50/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_50/MatMul/ReadVariableOpReadVariableOp+lstm_cell_50_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_50/MatMulMatMulstrided_slice_2:output:0*lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_50_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_50/MatMul_1MatMulzeros:output:0,lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_50/addAddV2lstm_cell_50/MatMul:product:0lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_50_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_50/BiasAddBiasAddlstm_cell_50/add:z:0+lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_50/splitSplit%lstm_cell_50/split/split_dim:output:0lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_50/SigmoidSigmoidlstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_50/Sigmoid_1Sigmoidlstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_50/mulMullstm_cell_50/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_50/ReluRelulstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_50/mul_1Mullstm_cell_50/Sigmoid:y:0lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_50/add_1AddV2lstm_cell_50/mul:z:0lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_50/Sigmoid_2Sigmoidlstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_50/Relu_1Relulstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_50/mul_2Mullstm_cell_50/Sigmoid_2:y:0!lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_50_matmul_readvariableop_resource-lstm_cell_50_matmul_1_readvariableop_resource,lstm_cell_50_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_963161*
condR
while_cond_963160*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_50/BiasAdd/ReadVariableOp#^lstm_cell_50/MatMul/ReadVariableOp%^lstm_cell_50/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_50/BiasAdd/ReadVariableOp#lstm_cell_50/BiasAdd/ReadVariableOp2H
"lstm_cell_50/MatMul/ReadVariableOp"lstm_cell_50/MatMul/ReadVariableOp2L
$lstm_cell_50/MatMul_1/ReadVariableOp$lstm_cell_50/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
Ã
while_cond_961803
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_961803___redundant_placeholder04
0while_while_cond_961803___redundant_placeholder14
0while_while_cond_961803___redundant_placeholder24
0while_while_cond_961803___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
È	
ö
E__inference_dense_203_layer_call_and_return_conditional_losses_963284

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Â	
¯
/__inference_sequential_101_layer_call_fn_962210
lstm_50_input
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCalllstm_50_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_101_layer_call_and_return_conditional_losses_962174o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_50_input
µ
Ã
while_cond_961660
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_961660___redundant_placeholder04
0while_while_cond_961660___redundant_placeholder14
0while_while_cond_961660___redundant_placeholder24
0while_while_cond_961660___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
¨J

C__inference_lstm_50_layer_call_and_return_conditional_losses_961888

inputs>
+lstm_cell_50_matmul_readvariableop_resource:	@
-lstm_cell_50_matmul_1_readvariableop_resource:	 ;
,lstm_cell_50_biasadd_readvariableop_resource:	
identity¢#lstm_cell_50/BiasAdd/ReadVariableOp¢"lstm_cell_50/MatMul/ReadVariableOp¢$lstm_cell_50/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_50/MatMul/ReadVariableOpReadVariableOp+lstm_cell_50_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_50/MatMulMatMulstrided_slice_2:output:0*lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_50_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_50/MatMul_1MatMulzeros:output:0,lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_50/addAddV2lstm_cell_50/MatMul:product:0lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_50_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_50/BiasAddBiasAddlstm_cell_50/add:z:0+lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_50/splitSplit%lstm_cell_50/split/split_dim:output:0lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitn
lstm_cell_50/SigmoidSigmoidlstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_50/Sigmoid_1Sigmoidlstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_cell_50/mulMullstm_cell_50/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ h
lstm_cell_50/ReluRelulstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_50/mul_1Mullstm_cell_50/Sigmoid:y:0lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
lstm_cell_50/add_1AddV2lstm_cell_50/mul:z:0lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
lstm_cell_50/Sigmoid_2Sigmoidlstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
lstm_cell_50/Relu_1Relulstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_50/mul_2Mullstm_cell_50/Sigmoid_2:y:0!lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_50_matmul_readvariableop_resource-lstm_cell_50_matmul_1_readvariableop_resource,lstm_cell_50_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_961804*
condR
while_cond_961803*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
NoOpNoOp$^lstm_cell_50/BiasAdd/ReadVariableOp#^lstm_cell_50/MatMul/ReadVariableOp%^lstm_cell_50/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_50/BiasAdd/ReadVariableOp#lstm_cell_50/BiasAdd/ReadVariableOp2H
"lstm_cell_50/MatMul/ReadVariableOp"lstm_cell_50/MatMul/ReadVariableOp2L
$lstm_cell_50/MatMul_1/ReadVariableOp$lstm_cell_50/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
µ
(__inference_lstm_50_layer_call_fn_962673

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_50_layer_call_and_return_conditional_losses_962123o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù"
ã
while_body_961470
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_50_961494_0:	.
while_lstm_cell_50_961496_0:	 *
while_lstm_cell_50_961498_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_50_961494:	,
while_lstm_cell_50_961496:	 (
while_lstm_cell_50_961498:	¢*while/lstm_cell_50/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0³
*while/lstm_cell_50/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_50_961494_0while_lstm_cell_50_961496_0while_lstm_cell_50_961498_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_50_layer_call_and_return_conditional_losses_961456Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_50/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity3while/lstm_cell_50/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/Identity_5Identity3while/lstm_cell_50/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y

while/NoOpNoOp+^while/lstm_cell_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_50_961494while_lstm_cell_50_961494_0"8
while_lstm_cell_50_961496while_lstm_cell_50_961496_0"8
while_lstm_cell_50_961498while_lstm_cell_50_961498_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_50/StatefulPartitionedCall*while/lstm_cell_50/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
­	
¨
/__inference_sequential_101_layer_call_fn_962277

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2:  
	unknown_3: 
	unknown_4: 
	unknown_5:
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_101_layer_call_and_return_conditional_losses_961930o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ö
E__inference_dense_202_layer_call_and_return_conditional_losses_961907

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
8
Ð
while_body_963161
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_50_matmul_readvariableop_resource_0:	H
5while_lstm_cell_50_matmul_1_readvariableop_resource_0:	 C
4while_lstm_cell_50_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_50_matmul_readvariableop_resource:	F
3while_lstm_cell_50_matmul_1_readvariableop_resource:	 A
2while_lstm_cell_50_biasadd_readvariableop_resource:	¢)while/lstm_cell_50/BiasAdd/ReadVariableOp¢(while/lstm_cell_50/MatMul/ReadVariableOp¢*while/lstm_cell_50/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_50/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_50_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_50/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
*while/lstm_cell_50/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¡
while/lstm_cell_50/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_50/addAddV2#while/lstm_cell_50/MatMul:product:0%while/lstm_cell_50/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_50/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_50_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_50/BiasAddBiasAddwhile/lstm_cell_50/add:z:01while/lstm_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_50/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_50/splitSplit+while/lstm_cell_50/split/split_dim:output:0#while/lstm_cell_50/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitz
while/lstm_cell_50/SigmoidSigmoid!while/lstm_cell_50/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_50/Sigmoid_1Sigmoid!while/lstm_cell_50/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mulMul while/lstm_cell_50/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
while/lstm_cell_50/ReluRelu!while/lstm_cell_50/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mul_1Mulwhile/lstm_cell_50/Sigmoid:y:0%while/lstm_cell_50/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/add_1AddV2while/lstm_cell_50/mul:z:0while/lstm_cell_50/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
while/lstm_cell_50/Sigmoid_2Sigmoid!while/lstm_cell_50/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
while/lstm_cell_50/Relu_1Reluwhile/lstm_cell_50/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_50/mul_2Mul while/lstm_cell_50/Sigmoid_2:y:0'while/lstm_cell_50/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_50/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_50/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
while/Identity_5Identitywhile/lstm_cell_50/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð

while/NoOpNoOp*^while/lstm_cell_50/BiasAdd/ReadVariableOp)^while/lstm_cell_50/MatMul/ReadVariableOp+^while/lstm_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_50_biasadd_readvariableop_resource4while_lstm_cell_50_biasadd_readvariableop_resource_0"l
3while_lstm_cell_50_matmul_1_readvariableop_resource5while_lstm_cell_50_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_50_matmul_readvariableop_resource3while_lstm_cell_50_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2V
)while/lstm_cell_50/BiasAdd/ReadVariableOp)while/lstm_cell_50/BiasAdd/ReadVariableOp2T
(while/lstm_cell_50/MatMul/ReadVariableOp(while/lstm_cell_50/MatMul/ReadVariableOp2X
*while/lstm_cell_50/MatMul_1/ReadVariableOp*while/lstm_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
÷
µ
(__inference_lstm_50_layer_call_fn_962662

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_50_layer_call_and_return_conditional_losses_961888o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
Ã
while_cond_962731
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_962731___redundant_placeholder04
0while_while_cond_962731___redundant_placeholder14
0while_while_cond_962731___redundant_placeholder24
0while_while_cond_962731___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
µ
Ã
while_cond_962874
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_962874___redundant_placeholder04
0while_while_cond_962874___redundant_placeholder14
0while_while_cond_962874___redundant_placeholder24
0while_while_cond_962874___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
º

(sequential_101_lstm_50_while_cond_961291J
Fsequential_101_lstm_50_while_sequential_101_lstm_50_while_loop_counterP
Lsequential_101_lstm_50_while_sequential_101_lstm_50_while_maximum_iterations,
(sequential_101_lstm_50_while_placeholder.
*sequential_101_lstm_50_while_placeholder_1.
*sequential_101_lstm_50_while_placeholder_2.
*sequential_101_lstm_50_while_placeholder_3L
Hsequential_101_lstm_50_while_less_sequential_101_lstm_50_strided_slice_1b
^sequential_101_lstm_50_while_sequential_101_lstm_50_while_cond_961291___redundant_placeholder0b
^sequential_101_lstm_50_while_sequential_101_lstm_50_while_cond_961291___redundant_placeholder1b
^sequential_101_lstm_50_while_sequential_101_lstm_50_while_cond_961291___redundant_placeholder2b
^sequential_101_lstm_50_while_sequential_101_lstm_50_while_cond_961291___redundant_placeholder3)
%sequential_101_lstm_50_while_identity
¾
!sequential_101/lstm_50/while/LessLess(sequential_101_lstm_50_while_placeholderHsequential_101_lstm_50_while_less_sequential_101_lstm_50_strided_slice_1*
T0*
_output_shapes
: y
%sequential_101/lstm_50/while/IdentityIdentity%sequential_101/lstm_50/while/Less:z:0*
T0
*
_output_shapes
: "W
%sequential_101_lstm_50_while_identity.sequential_101/lstm_50/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¼
serving_default¨
K
lstm_50_input:
serving_default_lstm_50_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_2030
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:´s
Û
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ú
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
Ñ
&iter

'beta_1

(beta_2
	)decay
*learning_ratemVmWmXmY+mZ,m[-m\v]v^v_v`+va,vb-vc"
	optimizer
Q
+0
,1
-2
3
4
5
6"
trackable_list_wrapper
Q
+0
,1
-2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
2
/__inference_sequential_101_layer_call_fn_961947
/__inference_sequential_101_layer_call_fn_962277
/__inference_sequential_101_layer_call_fn_962296
/__inference_sequential_101_layer_call_fn_962210À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ö2ó
J__inference_sequential_101_layer_call_and_return_conditional_losses_962452
J__inference_sequential_101_layer_call_and_return_conditional_losses_962608
J__inference_sequential_101_layer_call_and_return_conditional_losses_962231
J__inference_sequential_101_layer_call_and_return_conditional_losses_962252À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÒBÏ
!__inference__wrapped_model_961389lstm_50_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
3serving_default"
signature_map
ø
4
state_size

+kernel
,recurrent_kernel
-bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9_random_generator
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

<states
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
(__inference_lstm_50_layer_call_fn_962640
(__inference_lstm_50_layer_call_fn_962651
(__inference_lstm_50_layer_call_fn_962662
(__inference_lstm_50_layer_call_fn_962673Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
C__inference_lstm_50_layer_call_and_return_conditional_losses_962816
C__inference_lstm_50_layer_call_and_return_conditional_losses_962959
C__inference_lstm_50_layer_call_and_return_conditional_losses_963102
C__inference_lstm_50_layer_call_and_return_conditional_losses_963245Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
":   2dense_202/kernel
: 2dense_202/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_202_layer_call_fn_963254¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_202_layer_call_and_return_conditional_losses_963265¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
":  2dense_203/kernel
:2dense_203/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_203_layer_call_fn_963274¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_203_layer_call_and_return_conditional_losses_963284¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	2lstm_50/lstm_cell_50/kernel
8:6	 2%lstm_50/lstm_cell_50/recurrent_kernel
(:&2lstm_50/lstm_cell_50/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
L0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÑBÎ
$__inference_signature_wrapper_962629lstm_50_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
5	variables
6trainable_variables
7regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
¢2
-__inference_lstm_cell_50_layer_call_fn_963301
-__inference_lstm_cell_50_layer_call_fn_963318¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
H__inference_lstm_cell_50_layer_call_and_return_conditional_losses_963350
H__inference_lstm_cell_50_layer_call_and_return_conditional_losses_963382¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Rtotal
	Scount
T	variables
U	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
R0
S1"
trackable_list_wrapper
-
T	variables"
_generic_user_object
':%  2Adam/dense_202/kernel/m
!: 2Adam/dense_202/bias/m
':% 2Adam/dense_203/kernel/m
!:2Adam/dense_203/bias/m
3:1	2"Adam/lstm_50/lstm_cell_50/kernel/m
=:;	 2,Adam/lstm_50/lstm_cell_50/recurrent_kernel/m
-:+2 Adam/lstm_50/lstm_cell_50/bias/m
':%  2Adam/dense_202/kernel/v
!: 2Adam/dense_202/bias/v
':% 2Adam/dense_203/kernel/v
!:2Adam/dense_203/bias/v
3:1	2"Adam/lstm_50/lstm_cell_50/kernel/v
=:;	 2,Adam/lstm_50/lstm_cell_50/recurrent_kernel/v
-:+2 Adam/lstm_50/lstm_cell_50/bias/v¡
!__inference__wrapped_model_961389|+,-:¢7
0¢-
+(
lstm_50_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_203# 
	dense_203ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_202_layer_call_and_return_conditional_losses_963265\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 }
*__inference_dense_202_layer_call_fn_963254O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¥
E__inference_dense_203_layer_call_and_return_conditional_losses_963284\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_203_layer_call_fn_963274O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿÄ
C__inference_lstm_50_layer_call_and_return_conditional_losses_962816}+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 Ä
C__inference_lstm_50_layer_call_and_return_conditional_losses_962959}+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ´
C__inference_lstm_50_layer_call_and_return_conditional_losses_963102m+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ´
C__inference_lstm_50_layer_call_and_return_conditional_losses_963245m+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
(__inference_lstm_50_layer_call_fn_962640p+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
(__inference_lstm_50_layer_call_fn_962651p+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
(__inference_lstm_50_layer_call_fn_962662`+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
(__inference_lstm_50_layer_call_fn_962673`+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ Ê
H__inference_lstm_cell_50_layer_call_and_return_conditional_losses_963350ý+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 Ê
H__inference_lstm_cell_50_layer_call_and_return_conditional_losses_963382ý+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 
-__inference_lstm_cell_50_layer_call_fn_963301í+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ 
-__inference_lstm_cell_50_layer_call_fn_963318í+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ Â
J__inference_sequential_101_layer_call_and_return_conditional_losses_962231t+,-B¢?
8¢5
+(
lstm_50_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
J__inference_sequential_101_layer_call_and_return_conditional_losses_962252t+,-B¢?
8¢5
+(
lstm_50_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 »
J__inference_sequential_101_layer_call_and_return_conditional_losses_962452m+,-;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 »
J__inference_sequential_101_layer_call_and_return_conditional_losses_962608m+,-;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_sequential_101_layer_call_fn_961947g+,-B¢?
8¢5
+(
lstm_50_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_101_layer_call_fn_962210g+,-B¢?
8¢5
+(
lstm_50_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_101_layer_call_fn_962277`+,-;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_101_layer_call_fn_962296`+,-;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¶
$__inference_signature_wrapper_962629+,-K¢H
¢ 
Aª>
<
lstm_50_input+(
lstm_50_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_203# 
	dense_203ÿÿÿÿÿÿÿÿÿ