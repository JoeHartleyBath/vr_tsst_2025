import sys
fn = sys.argv[1] if len(sys.argv)>1 else 'tmpdata61863.fdt'
with open(fn,'rb') as f:
    b=f.read(64)
print(' '.join('{:02X}'.format(x) for x in b))
