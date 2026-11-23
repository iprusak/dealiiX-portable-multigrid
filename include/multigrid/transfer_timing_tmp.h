// TEMPORARY instrumentation (transfer weights timing), not to be committed.
#ifndef transfer_timing_tmp_h
#define transfer_timing_tmp_h

#include <deal.II/base/mpi.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace TransferTimingTmp
{
  // one line: tag key=value ..., times as maximum over the ranks
  inline void
  print(const MPI_Comm                                     comm,
        const std::string                                 &tag,
        const std::string                                 &info,
        const std::vector<std::pair<std::string, double>> &times)
  {
    std::vector<double> values;
    for (const auto &t : times)
      values.push_back(t.second);
    std::vector<double> max_values(values.size());
    dealii::Utilities::MPI::max(values, comm, max_values);
    if (dealii::Utilities::MPI::this_mpi_process(comm) == 0)
      {
        std::cout << "TWT " << tag << ' ' << info;
        for (unsigned int i = 0; i < times.size(); ++i)
          std::cout << ' ' << times[i].first << '=' << std::scientific << std::setprecision(4)
                    << max_values[i];
        std::cout << std::defaultfloat << std::endl;
      }
  }
} // namespace TransferTimingTmp

#endif
