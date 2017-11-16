import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate as interp

import ase
from ase.calculators import lj
from ase import build

from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.io.trajectory import Trajectory

def get_lj_fcc_struc(super_a=[4,4,4],dNN=1.0,E_lj=1/40):
    """
    dNN in units of LJ equilibrium distance
      -> NOT standard LJ units of sigma
    """

    # amin is unitcell value required for a nearest neighbor distance
    #  equal to the equilibrium point of LJ potential
    amin=np.sqrt(2)
    a = dNN*amin
    fcc_at = ase.Atoms(scaled_positions=[[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]],cell=[a,a,a],
                      pbc=True)
    xtal_at = build.make_supercell(fcc_at,P=super_a*np.eye(3))
    # Set depth equal to 1/40 eV, or 300 K equiv
    xtal_at.set_calculator(lj.LennardJones(epsilon=E_lj))

    return xtal_at

def run_mdsim(liq_at,T,Nstep,dt=0.1*units.fs,Tdampfac=100,rand_vel=False,
              traj_filenm=None,traj_freq=None,traj_append=False):

    assert np.all(liq_at.get_pbc()), 'Simbox must have Periodic Boundary Conditions'

    if traj_filenm is not None:
        if traj_append:
            mode = 'a'
        else:
            mode = 'w'

        traj = Trajectory(traj_filenm, mode, liq_at)

        if traj_freq is None:
            traj_freq = np.ceil(.1*Nstep)

    if rand_vel:
        kT = T*units.kB
        MaxwellBoltzmannDistribution(liq_at, kT )



    # Set the momenta corresponding to T=300K

    # We want to run MD with constant energy using the VelocityVerlet algorithm.
    # dyn = VelocityVerlet(liq_at, dt=1e-3)

    # Should have value near 100
    Tdamp = Tdampfac*dt
    dyn = NVTBerendsen(liq_at, dt, T, Tdamp)

    if traj_filenm is not None:
        dyn.attach(traj.write, interval=traj_freq)

    dyn.run(Nstep)

    if traj_filenm is not None:
        traj.close()

    pass


def calc_pdf( liq_in_at, dist_range=[.5,3], nbins=50 ):
    # liq_at = build.make_supercell(liq_in_at,P=[4,4,4]*np.eye(3))
    liq_at = liq_in_at
    assert np.all(liq_at.get_pbc()), 'Simbox must have Periodic Boundary Conditions'

    all_dist_a = liq_at.get_all_distances(mic=True)
    Nat = liq_at.get_number_of_atoms()
    Vat = liq_at.get_volume()

    dists_a = np.sort(all_dist_a.ravel())[Nat:]


    # hist_a, bin_edges_a = np.histogram(all_dist_a,
    #                                    range=grange,
    #                                    bins=bins)
    # plt.plot(dists_a,'k-')
    # hist_a, bin_edges_a = np.histogram(dists_a,bins=bins,density=False)
    hist_a, bin_edges_a = np.histogram(dists_a,bins=nbins,range=dist_range,density=False)
    Npair_cnt = np.sum(hist_a)

    # print(hist_a)
    # print(np.sum(hist_a))
    # print(Ncnt)
    # print(Nat)
    rhoavg = Nat/Vat

    dr = np.diff(bin_edges_a)[0]
    r_a = bin_edges_a[0:-1]+dr/2
    dV_a = 4*np.pi*dr*r_a**2
    dN_a = hist_a/Nat
    g_a = dN_a/dV_a/rhoavg

    # ind_a = np.arange(len(r_a)+1)
    dV_a = 4/3*np.pi*(bin_edges_a[1:]**3-bin_edges_a[:-1]**3)
    g_a = hist_a/(Nat*(Nat-1))*(Vat/dV_a)
    # g_a = hist_a/(Npair_cnt)*(Vat/dV_a)
    # g_a = hist_a/(Nat**2)*(Vat/dV_a)

    return g_a, r_a

def get_traj_data(traj_filenm,store_atoms=True):
    traj = Trajectory(traj_filenm)
    Etot_a = np.zeros(len(traj))
    Epot_a = np.zeros(len(traj))
    Ekin_a = np.zeros(len(traj))
    iter_a = np.zeros(len(traj))
    atoms_l = []
    for ind,atoms in enumerate(traj):
        Etot_a[ind] = atoms.get_total_energy()
        Epot_a[ind] = atoms.get_potential_energy()
        Ekin_a[ind] = atoms.get_kinetic_energy()
        iter_a[ind] = ind
        atoms_l.append(atoms)


    traj_d = {}
    traj_d['Etot'] = Etot_a
    traj_d['Epot'] = Epot_a
    traj_d['Ekin'] = Ekin_a
    traj_d['iter'] = iter_a

    if store_atoms:
        traj_d['atoms'] = atoms_l


    traj.close()

    return traj_d

def plot_energy_convergence(traj_filenm,istart=0,dist_range=[.5,3]):
    traj_d = get_traj_data(traj_filenm,store_atoms=True)

    atoms_fin = traj_d['atoms'][-1]
    Nat = atoms_fin.get_number_of_atoms()
    plt.figure()
    plt.plot(traj_d['iter'][istart:],traj_d['Ekin'][istart:]/(Nat*3/2*units.kB),'bo-')
    plt.xlabel('iter')
    plt.ylabel('Dynamical Temp [K]')

    plt.figure()
    plt.plot(traj_d['iter'][istart:],traj_d['Epot'][istart:]/(Nat*3/2*units.kB),'bo-')
    plt.xlabel('iter')
    plt.ylabel('Potential Energy [K equiv]')


    plt.figure()
    g_a, r_a = calc_pdf( atoms_fin, dist_range=dist_range, nbins=100 )
    plt.plot(r_a,g_a,'ko-')
    plt.title('Final Structure')
    pass


def calc_avg_pdf(traj_filenm, istart=0,dist_range = [0.5,3.0],nbins=100):
    traj_d = get_traj_data(traj_filenm,store_atoms=True)
    g_all_a = []

    for atoms in traj_d['atoms'][istart:]:
        ig_a, r_a = calc_pdf( atoms, dist_range=dist_range, nbins=nbins )
        g_all_a.append(ig_a)

    g_all_a = np.array(g_all_a)
    g_avg_a = np.mean(g_all_a,axis=0)

    return r_a, g_avg_a


def calc_coor_num(rho,r_dat_a,g_dat_a,debug=False,Ngrid=1001,rcut=1.5):
    """
    Calculate coordination number by integrating pair distribution function.

    Integrate up to rcut located at first min in the pair distribution function (pdf)

    Parameters
    ----------
    rho : double
        average density of the simulation.
    r_dat_a : array
        grid of distances from central atom.
    g_dat_a : array
        pair distribution function (rho/avgrho) for central atom.
    debug : boolean, default False
        optional flag return debug dictionary if True (see below).
    Ngrid : integer
        number of resolution grid points for the r values.

    Returns
    -------
    cn : double
        average number of bonded neighbors for central atom
        (doesn't have to be integer because it's an average).
    debug_d : dictionary (optional)
        if debug is True, return dictionary of intermediate
        values with keys:
        ``imin`` : int
            index of fist min in the pair distribution function array.
        ``rcut`` : double
            distance at the first min defining boundary of first shell
            in the pair distribution funtion. upper limit of integration.
        ``dn_atom_a`` : double array
            average number of atoms in the spherical shell around central atom.
        ``r_a`` : double array
            distance from central atom gridded to high resolution.
        ``g_a`` : double array
            pair distribution function of simulation interpolated to high resolution.


    """

    r_a = np.linspace(r_dat_a[0],r_dat_a[-1],Ngrid)
    g_a = interp.interp1d(r_dat_a,g_dat_a,kind='cubic')(r_a)


    indmax = np.argmax(g_a)
    indmin = indmax-1+np.argmin(g_a[indmax:])
    dn_atom = 4*np.pi*r_a**2*g_a*rho
    cn = np.trapz(dn_atom[0:indmin+1],x=r_a[0:indmin+1])

    if not debug:
        return cn
    else:
        debug_d = {}
        debug_d['imin'] = indmin
        debug_d['rmical'] = r_a[indmin]
        debug_d['dn_a'] = dn_atom
        debug_d['r_a'] = r_a
        debug_d['g_a'] = g_a
        return cn, debug_d

def calc_rcut_global(pdf_a,sampdist_a):

    ind_max = np.argmax(pdf_a)
    ind_min= np.argmin(pdf_a[ind_max:]) + ind_max
    rcut = sampdist_a[ind_min]

    return rcut

def calc_rcut_local(pdf_a,sampdist_a):

    ind_max = np.argmax(pdf_a)
    pdf_diff_a = np.diff(pdf_a[ind_max:])
    ind_min= np.where(pdf_diff_a> 0)[0][0] + ind_max


    #ind_min= np.argmin(pdf_a[ind_max:]) + ind_max
    rcut = sampdist_a[ind_min]

    return rcut

def coor_cutoff(dists_a,rcut=1.5):
    assert dists_a.ndim==1, 'dists_a must be a 1 dimensional array'

    # inputs distances and returns coordination number
    w_a = dists_a <= rcut
    coornum = np.sum(w_a)
    return coornum

def coor_econ(dists_a,TOL=1e-5,dosort=True,nexp=6):
    # assert len(dists_a.shape)
    assert dists_a.ndim==1, 'dists_a must be a 1 dimensional array'


    if dosort:
        dists_a = np.sort(dists_a)

    if dists_a[0]==0:
        dists_a = dists_a[1:]

    dnnavg = dists_a[0]

    # loops until distances stop changing
    while True:
        w_a = np.exp(1 - (dists_a / dnnavg)**nexp)
        dnnavg_next = np.sum(w_a * dists_a) / np.sum(w_a)
        if np.abs(dnnavg_next/dnnavg-1) < TOL:
            break
        dnnavg = dnnavg_next

    coornum = np.sum(w_a)
    return coornum

def coor_sann(idists_a):
    assert idists_a.ndim==1, 'dists_a must be a 1 dimensional array'

    N = idists_a.shape[0]
    rad1=idists_a[0]
    Nbond = 3
    # Nbond bc we want the 3rd index to be the 4th atom(the 1st atom that's not bonded)
    rnonbond = idists_a[Nbond]
    # sum over Nbond neighbors
    rshell = np.sum(idists_a[:Nbond]/(Nbond-2))
    while rshell > rnonbond:
        Nbond += 1
        rshell = np.sum(idists_a[:Nbond]/(Nbond-2))
        rnonbond = idists_a[Nbond]


    w_a = (1-idists_a[:Nbond]/rshell)/(1-rad1/rshell)
    coornum = np.sum(w_a)
    return coornum

def local_kernel(x_a):
    q_a = np.abs(x_a)
    kernel_a = np.zeros(q_a.shape)
    ind_outer = np.where(np.abs(q_a)<=2)
    kernel_a[ind_outer] = 1/6*(2-q_a[ind_outer])**3

    ind_inner = np.where(np.abs(q_a)<=1)
    kernel_a[ind_inner] = 1/6*( (2-q_a[ind_inner])**3 - 4*(1-q_a[ind_inner])**3 )

    return kernel_a

def approx_pdf_kde(iatom, dists_a,rhobar,bandwidth=0.25):
    Nx = 5
    liq_at = get_lj_fcc_struc(super_a=[Nx,Nx,Nx],dNN=1.2,E_lj=1/40)


    Nat = liq_at.get_number_of_atoms()
    Vat = liq_at.get_volume()
    Lat = (Vat)**(1/3)
    rhoat = Nat/Vat

    if dists_a[0] == 0:
        dists_a = dists_a[1:]
    #FIX
    Nsamp = 1001
    pdf_approx_a = np.zeros(Nsamp)
    sampdist_a = np.linspace(0.001, Lat/2, Nsamp)
    for ind in np.arange(dists_a.size):
        #calculates the observed number of atoms per angstrom
        #if integrated, coordination number is given
        # dn_dr_a = (1/np.sqrt(2*np.pi)/bandwidth)*np.exp((-1/2)*(dists_a[ind]-sampdist_a)**2/(bandwidth**2))
        dn_dr_a = local_kernel( (dists_a[ind]-sampdist_a)/bandwidth )/ bandwidth
        assert dn_dr_a[0] == 0, 'bandwidth is too large for the closest neighboring atom'
        g_a = dn_dr_a/(4*np.pi*sampdist_a**2*rhobar)
        #print(g_a)

        # print(dn_dr_a[:20])
        pdf_approx_a += g_a

    return pdf_approx_a, sampdist_a

def approx_pdf_kde_adapt(iatom, dists_a,rhobar,num_neighbor=4,bandwidth_fac=1):
    Nx = 5
    liq_at = get_lj_fcc_struc(super_a=[Nx,Nx,Nx],dNN=1.2,E_lj=1/40)


    Nat = liq_at.get_number_of_atoms()
    Vat = liq_at.get_volume()
    Lat = (Vat)**(1/3)
    rhoat = Nat/Vat

    if dists_a[0] == 0:
        dists_a = dists_a[1:]
    #FIX
    #bandwidth_a = np.array(idists_a.size)
    dist_diff_a = np.diff(dists_a)
    w_a=np.ones(num_neighbor-1)/(num_neighbor-1)
    avg_bandwidth_a=np.convolve(dist_diff_a,w_a, mode='valid')
    front_bandwidth_a=avg_bandwidth_a[0]*np.ones(num_neighbor/2)
    back_bandwidth_a =avg_bandwidth_a[-1]*np.ones(num_neighbor/2)
    bandwidth_a=bandwidth_fac*np.hstack((front_bandwidth_a,avg_bandwidth_a,back_bandwidth_a))*(dists_a/dists_a[0])**2
    print(bandwidth_a.shape)
    print(avg_bandwidth_a.shape)
    print(dists_a.shape)
    #print(bandwidth_a)

    dist_cutoff = Lat/2
    Nsamp = 1001
    pdf_approx_a = np.zeros(Nsamp)
    dn_dr_tot_a = np.zeros(Nsamp)
    sampdist_a = np.linspace(0.001, .8*dist_cutoff, Nsamp)

    #for ind in np.arange(np.round(idists_a.size*.8)):
    for ind in np.arange(idists_a.size):
        #calculates the observed number of atoms per angstrom
        #if integrated, coordination number is given
        ##dn_dr_a = (1/np.sqrt(2*np.pi)/bandwidth)*np.exp((-1/2)*(dists_a[ind]-sampdist_a)**2/(bandwidth**2))
        if dists_a[ind] > dist_cutoff:
            break
        ibandwidth= bandwidth_a[ind]
        dn_dr_a = local_kernel( (dists_a[ind]-sampdist_a)/ibandwidth )/ ibandwidth
        g_a = dn_dr_a/(4*np.pi*sampdist_a**2*rhobar)
        #print(g_a)

        # print(dn_dr_a[:20])
        pdf_approx_a += g_a
        dn_dr_tot_a +=dn_dr_a

    return pdf_approx_a, sampdist_a,dn_dr_tot_a
