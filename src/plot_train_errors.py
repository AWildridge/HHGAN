import numpy as np

from ROOT import TGraph, TLegend, TCanvas

FINAL_EPOCH = 10000

train_error = np.load('./Train_Record.npy')
wasserstein_estimate = -(train_error[2] + train_error[3])

canvas = TCanvas('', '', 1400, 1400)
gr_g_loss = TGraph(len(train_error[0][:FINAL_EPOCH]), train_error[0][:FINAL_EPOCH], train_error[1][:FINAL_EPOCH])
gr_wass_est = TGraph(len(train_error[0][:FINAL_EPOCH]), train_error[0][:FINAL_EPOCH], wasserstein_estimate[:FINAL_EPOCH])
gr_d_gp_loss = TGraph(len(train_error[0][:FINAL_EPOCH]), train_error[0][:FINAL_EPOCH], train_error[4][:FINAL_EPOCH])
loss_legend = TLegend(0.3, 0.15, 0.85, 0.4)
gr_wass_est.SetTitle('Loss from WGAN-GP')
gr_wass_est.GetXaxis().SetTitle('Generator Iterations')
gr_wass_est.GetYaxis().SetTitle('Loss')
gr_g_loss.SetLineColor(1)
gr_wass_est.SetLineColor(2)
gr_d_gp_loss.SetLineColor(4)
#gr_g_loss.SetLineWidth(2)
#gr_wass_est.SetLineWidth(2)
#gr_d_gp_loss.SetLineWidth(2)
loss_legend.AddEntry(gr_g_loss, 'Generator loss')
loss_legend.AddEntry(gr_wass_est, 'Wasserstein Estimate')
loss_legend.AddEntry(gr_d_gp_loss, 'Gradient Penalty Loss')
gr_g_loss.SetMinimum(-0.6)
gr_wass_est.Draw('AC')
#gr_g_loss.Draw('AC SAME')
#gr_d_gp_loss.Draw('C SAME')
loss_legend.Draw()

canvas.SaveAs('./training_error_plot.pdf')
canvas.SaveAs('./training_error_plot.root')
