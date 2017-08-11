
#include "EulerTwo.h"



int main( int argc, char *argv[] )
{
    // That is, there must be 8 or 9 argumets arguments.
    if (argc < 8)
	{
		cout << "The Program takes 8 inputs, #Divisions, #Threads/block, deltat, finish time, output frequency..." << endl;
        cout << "Algorithm type, Variable Output File, Timing Output File (optional)" << endl;
		exit(-1);
	}
    cout.precision(10); 

	// Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);
    if (sizeof(REAL)>6) cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    dimz.gam = 1.4;
    dimz.mgam = 0.4;

    bd[0].x = ONE; //Density
    bd[1].x = 0.125;
    bd[0].y = ZERO; //Velocity
    bd[1].y = ZERO;
    bd[0].z = ONE/dimz.mgam; //Energy
    bd[1].z = 0.1/dimz.mgam;

    const int dv = atoi(argv[1]); //Number of spatial points
	const int tpb = atoi(argv[2]); //Threads per Block
    const double dt = atof(argv[3]); //Timestep
	const double tf = atof(argv[4]) - QUARTER*dt; //Finish time
    const double freq = atof(argv[5]); //Frequency of output (i.e. every 20 s (simulation time))
    const int scheme = atoi(argv[6]); //2 for Alternate, 1 for GPUShared, 0 for Classic
    const int bks = dv/tpb; //The number of blocks
    const double dx = lx/((REAL)dv-TWO); //Grid size.
    char const *prec;
    prec = (sizeof(REAL)<6) ? "Single": "Double";

    //Declare the dimensions in constant memory.
    dimz.dt_dx = dt/dx; // dt/dx
    dimz.base = tpb+4; // Length of the base of a node.
    dimz.idxend = dv-1; // Index of last spatial point.
    dimz.idxend_1 = dv-2; // 2nd to last spatial point.

    for (int k=-2; k<3; k++) dimz.hts[k+2] = (tpb/2) + k; //Middle values in the node (masking values)

    cout << "Euler --- #Blocks: " << bks << " | Length: " << lx << " | Precision: " << prec << " | dt/dx: " << dimz.dt_dx << endl;

	// Conditions for main input.
	// dv and tpb must be powers of two.  dv must be larger than tpb and divisible by tpb.

	if ((dv & (tpb-1) !=0) || (tpb&31) != 0)
    {
        cout << "INVALID NUMERIC INPUT!! "<< endl;
        cout << "2nd ARGUMENT MUST BE A POWER OF TWO >= 32 AND FIRST ARGUMENT MUST BE DIVISIBLE BY SECOND" << endl;
        exit(-1);
    }

    if (dimz.dt_dx > .21)
    {
        cout << "The value of dt/dx (" << dimz.dt_dx << ") is too high.  In general it must be <=.21 for stability." << endl;
        exit(-1);
    }

	// Initialize arrays.
    REALthree *IC, *T_final;
	cudaHostAlloc((void **) &IC, dv*sizeof(REALthree), cudaHostAllocDefault); // Initial conditions
	cudaHostAlloc((void **) &T_final, dv*sizeof(REALthree), cudaHostAllocDefault); // Final values

	for (int k = 0; k<dv; k++) IC[k] = (k<dv/2) ? bd[0] : bd[1]; // Populate initial conditions

	// Call out the file before the loop and write out the initial condition.
	ofstream fwr;
	fwr.open(argv[7],ios::trunc);
    fwr.precision(10);

	// Write out x length and then delta x and then delta t.
	// First item of each line is variable second is timestamp.
	fwr << lx << " " << (dv-2) << " " << dx << " " << endl;

    fwr << "Density " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].x << " ";
    fwr << endl;

    fwr << "Velocity " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].y << " ";
    fwr << endl;

    fwr << "Energy " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].z/IC[k].x << " ";
    fwr << endl;

    fwr << "Pressure " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << pressure(IC[k]) << " ";
    fwr << endl;

    // Transfer data to GPU in constant memory.
	cudaMemcpyToSymbol(dimens,&dimz,sizeof(dimensions));
    cudaMemcpyToSymbol(dbd,&bd,2*sizeof(REALthree));

    // Start the counter and start the clock.
    cudaEvent_t start, stop;
	float timed;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0);

    // Call the correct function with the correct algorithm.
    cout << scheme << " " ;
    double tfm;
    if (scheme)
    {
        tfm = sweptWrapper(bks, tpb, dv, dt, tf, scheme-1, IC, T_final, freq, fwr);
    }
    else
    {
        tfm = classicWrapper(bks, tpb, dv, dt, tf, IC, T_final, freq, fwr);
    }

	// Show the time and write out the final condition.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &timed, start, stop);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    timed *= 1.e3;

    double n_timesteps = tfm/dt;

    double per_ts = timed/n_timesteps;

    cout << n_timesteps << " timesteps" << endl;
	cout << "Averaged " << per_ts << " microseconds (us) per timestep" << endl;

    // Write out time per timestep to timing file.
    if (argc>7)
    {
        ofstream ftime;
        ftime.open(argv[8],ios::app);
    	ftime << dv << "\t" << tpb << "\t" << per_ts << endl;
    	ftime.close();
    }

	fwr << "Density " << tfm << " ";
	for (int k = 1; k<(dv-1); k++) fwr << T_final[k].x << " ";
    fwr << endl;

    fwr << "Velocity " << tfm << " ";
	for (int k = 1; k<(dv-1); k++) fwr << T_final[k].y/T_final[k].x << " ";
    fwr << endl;

    fwr << "Energy " << tfm << " ";
    for (int k = 1; k<(dv-1); k++) fwr << energy(T_final[k]) << " ";
    fwr << endl;

    fwr << "Pressure " << tfm << " ";
    for (int k = 1; k<(dv-1); k++) fwr << pressure(T_final[k]) << " ";
    fwr << endl;

	fwr.close();

    cudaDeviceSynchronize();

	cudaEventDestroy( start );
	cudaEventDestroy( stop );
    cudaDeviceReset();
    cudaFreeHost(IC);
    cudaFreeHost(T_final);

	return 0;
}