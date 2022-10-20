from statistics import ClassHistory
import fake_data as fd

mmsr_data = fd.mmsr_data
maintenance_periods_list = fd.maintenance_periods_list
deposit_rate_data = fd.deposit_rate_data


history = ClassHistory()
history.build_from_data(mmsr_data, deposit_rate_data, maintenance_periods_list)
