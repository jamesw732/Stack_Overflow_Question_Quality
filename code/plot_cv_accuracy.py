import matplotlib.pyplot as plt
import os

methods = ['Logistic Regression', 'SVM (Linear Kernel)', 'SVM (Polynomial Kernel)', 'SVM (RBF Kernel)', 'XGBoost']
cv_accuracies = [0.602946577461676, 0.5944411628794728, 0.5765724388724811, 0.6148658911458019, 0.875]
cv_accuracies = [100 * v for v in cv_accuracies]


fig, ax = plt.subplots()
ax.barh(methods, cv_accuracies, height=0.25, color='r')

ax.set_xlabel('Cross-Validation Accuracy (%)')
ax.set_ylabel('Features')

for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
ax.grid(color='grey',
        linestyle='-.', linewidth=0.5,
        alpha = 0.2)
plt.subplots_adjust(left=0.26)

savedir = os.path.abspath(os.path.join(os.path.realpath(__file__), 
                        '../../report/figures/cv_accuracies.png'))
plt.savefig(savedir, bbox_inches='tight')