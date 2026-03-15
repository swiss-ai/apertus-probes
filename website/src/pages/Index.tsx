import PaperLayout from "@/components/PaperLayout";
import FigureCard from "@/components/FigureCard";
import JsonChartLoader from "@/components/JsonChartLoader";
import LayerRMSEChart from "@/components/charts/LayerRMSEChart";
import TransferHeatmap from "@/components/charts/TransferHeatmap";
import CrossDatasetLineChart from "@/components/charts/CrossDatasetLineChart";
import MixedProbeLineChart from "@/components/charts/MixedProbeLineChart";
import SPIBarChart from "@/components/charts/SPIBarChart";
import SPIMixChart from "@/components/charts/SPIMixChart";
import SPIHeatmap from "@/components/charts/SPIHeatmap";
import "katex/dist/katex.min.css";
import { InlineMath, BlockMath } from "react-katex";

export default function Index() {
  return (
    <PaperLayout>
      {/* Header */}
      <header className="mb-12">
        <p className="font-sans text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-4">
          ETH Zurich · Data Science Lab · 2026
        </p>
        <h1 className="text-3xl md:text-4xl lg:text-[2.6rem] font-bold leading-tight tracking-tight font-sans mb-4">
          Detecting and Reducing Hallucination in Multiple-Choice Questions with Activation Steering
        </h1>
        <p className="text-lg text-muted-foreground leading-relaxed">
          We study mechanistic error reduction via activation steering on the Apertus family of language models, extending the MERA framework to mixed-domain and cross-dataset settings.
        </p>
        <div className="flex flex-wrap gap-x-4 gap-y-1 mt-4 font-sans text-sm text-muted-foreground">
          <span>Aleks Stepancic*</span>
          <span>Tu Nguyen*</span>

          <span>Eduard Durech</span>
          <span>Anna Hedström</span>
        </div>
        <p className="font-sans text-xs text-muted-foreground mt-1">* Equal contribution</p>
      </header>

      <article className="article-prose">
        {/* Abstract */}
        <section id="abstract">
          <h2>Abstract</h2>
          <p>
            Large language models can be unreliable even on simple multiple-choice tasks, and mitigating such errors at inference time remains challenging. This project studies mechanistic error reduction via activation steering on the Apertus family of language models, focusing on datasets where the base model exhibits low accuracy. We evaluate training probes and steering on single datasets and then extend prior work to mixed-dataset and cross-dataset settings. Across multiple datasets and models, we observe that linear probes on mid–late layers predict errors well, and a single probe trained on a mixture of all datasets performs competitively with per-dataset probes. Calibrated MERA-style steering improves accuracy across datasets and models, consistent with the original MERA results.
          </p>
        </section>

        {/* Motivation */}
        <section id="motivation">
          <h2>Motivation</h2>
          <p>
            Despite the considerable capabilities of modern language models, they remain error-prone in many tasks. Failures arise not only in open-ended settings such as reasoning, factual consistency, or planning, but also in simple prediction tasks and multiple-choice benchmarks. Reducing such errors remains an open and important research challenge.
          </p>
          <p>
            Recent work on Mechanistic Error Reduction with Abstention (MERA) proposes an inference-time framework for mitigating prediction errors. By acting on a single designated <strong>Layer L</strong>, this approach uses an <strong>Error Probe</strong> to estimate the likelihood of a mistake. Based on this prediction, it applies calibrated <strong>MERA Steering</strong> to safely intervene on the residual stream, abstaining when no reliable improvement can be guaranteed. This results in non-degrading performance while avoiding both under- and over-steering.
          </p>
          <p>
            In this work, we apply the MERA framework to the Apertus family of language models—a family where performance gaps on reasoning benchmarks have been reported. We extend the framework in two directions: <strong>mixed-domain probe training</strong>, where error-estimation probes are trained on heterogeneous datasets, and <strong>cross-dataset steering</strong>, where probes trained on one domain are used to steer on another.
          </p>
        </section>

        {/* Contributions */}
        <section id="contributions">
          <h2>Contributions</h2>
          <ul>
            <li>We reproduce the MERA framework on the Apertus model family, confirming that calibrated steering improves accuracy on datasets where the base model performs poorly.</li>
            <li>We train error-estimation probes on mixed-domain datasets and show they perform competitively with per-dataset probes, producing more robust error representations.</li>
            <li>We evaluate cross-dataset probe generalisation systematically, revealing structured but limited transfer between task domains.</li>
            <li>We adopt a normalised Steering Performance Impact (SPI) metric inspired by prior work that enables fair comparison across tasks with different baseline accuracies.</li>
          </ul>
        </section>

        {/* Problem Setup */}
        <section id="setup">
          <h2>Problem Setup</h2>
          <h3>Task</h3>
          <p>
            We consider autoregressive, decoder-only transformer language models evaluated on supervised multiple-choice and binary classification tasks. For each input prompt, the model produces hidden activations across layers. We evaluate logits at a selected token position: either the <em>last</em> token (final input token) or the <em>exact</em> token (first generated token matching a valid label).
          </p>
          <p>
            Given logits <InlineMath math="\mathbf{a}_k" /> at token position <InlineMath math="k" />, the predicted label <InlineMath math="\hat{y}" /> and its normalized probability are defined as:
          </p>
          <BlockMath math="\hat{y} = \arg\max_{y \in \mathcal{Y}} \mathbf{a}_{k,\text{idx}(y)}, \quad \text{prob}_{\hat{y}} = \frac{\exp(\mathbf{a}_{k,\text{idx}(\hat{y})})}{\sum_{y \in \mathcal{Y}} \exp(\mathbf{a}_{k,\text{idx}(y)})}" />
          <p>
            We define a continuous error signal:
          </p>
          <BlockMath math="E = 1 - \text{prob}_y" />
          <p>
            where <InlineMath math="\text{prob}_y" /> is the normalised probability assigned to the correct answer.
          </p>

          <h3>Models</h3>
          <p>
            We evaluate both base and instruction-tuned variants of two model families: <strong>Apertus-8B</strong> and <strong>Apertus-8B-Instruct</strong> (our primary focus), alongside <strong>Llama-3.1-8B</strong> and <strong>Llama-3.1-8B-Instruct</strong> for cross-family comparison. All models have 32 transformer layers.
          </p>

          <h3>Datasets</h3>
          <p>
            We evaluate on six benchmarks spanning reasoning, scientific knowledge, and text classification:
          </p>
          <div className="overflow-x-auto my-4">
            <table className="w-full font-sans text-sm border-collapse">
              <thead>
                <tr className="border-b-2">
                  <th className="text-left py-2 pr-4 font-semibold">Dataset</th>
                  <th className="text-left py-2 pr-4 font-semibold">Domain</th>
                  <th className="text-right py-2 font-semibold">Samples</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground">
                <tr className="border-b"><td className="py-1.5 pr-4">MMLU High School</td><td className="pr-4">MCQA / Reasoning</td><td className="text-right">3,790</td></tr>
                <tr className="border-b"><td className="py-1.5 pr-4">MMLU Professional</td><td className="pr-4">MCQA / Reasoning</td><td className="text-right">3,220</td></tr>
                <tr className="border-b"><td className="py-1.5 pr-4">ARC-Challenge</td><td className="pr-4">MCQA / Science</td><td className="text-right">2,590</td></tr>
                <tr className="border-b"><td className="py-1.5 pr-4">ARC-Easy</td><td className="pr-4">MCQA / Science</td><td className="text-right">4,626</td></tr>
                <tr className="border-b"><td className="py-1.5 pr-4">SMS Spam</td><td className="pr-4">Text Classification</td><td className="text-right">5,574</td></tr>
                <tr><td className="py-1.5 pr-4">Yes-No Finance</td><td className="pr-4">Finance</td><td className="text-right">5,000</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Method */}
        <section id="method">
          <h2>Method</h2>
          <p>
            Our pipeline consists of four stages: (1) activation extraction, (2) probe training, (3) steering strength calibration, and (4) inference-time steering. For each model and dataset, we cache hidden-state activations at every transformer layer during inference. Linear probes are then trained to predict model error from these activations. At inference time, the probe weights define steering directions, while a calibrated threshold determines when and how strongly to intervene.
          </p>

          <h3 id="probes">Probe Training</h3>
          <p>
            We train linear probes of the form <InlineMath math="\hat{p}(\mathbf{h}) = \mathbf{w}^\top \mathbf{h}" />, where <InlineMath math="\mathbf{h}" /> denotes activations at a given layer and token position. Probes are trained using supervised regression to predict the continuous error <InlineMath math="E(\mathbf{a})" />. We compare two probe types:
          </p>
          <ul>
            <li><strong>Linear probes (L-<InlineMath math="\alpha" />)</strong>: ridge regression with regularisation strength <InlineMath math="\alpha" />, directly regressing onto the error signal.</li>
            <li><strong>Logit probes (Logit-L-<InlineMath math="\alpha" />)</strong>: logistic regression mapping activations to a probability of error, with a subsequent logit transformation.</li>
          </ul>
          <p>
            Probe quality is measured using root mean squared error (RMSE) between predictions and ground-truth error. Lower RMSE indicates more accurate error estimation and a more reliable steering signal.
          </p>

          <h3 id="steering-methods">Steering Methods</h3>
          <p>We compare four inference-time steering strategies:</p>
          <ul>
            <li><strong>Prompt steering</strong>: task instructions injected into the input text. Serves as a no-access baseline.</li>
            <li><strong>Additive steering</strong>: fixed-strength addition of probe weights to activations.</li>
            <li><strong>Contrastive steering</strong>: difference-of-means direction between correct and incorrect examples.</li>
            <li><strong>MERA (calibrated steering)</strong>: optimally calibrated intervention that steers only when the predicted error exceeds a threshold <InlineMath math="\alpha" />, with strength proportional to the error magnitude. For linear probes, this yields a closed-form solution:</li>
          </ul>
          <BlockMath math="\mathbf{v}^\star = \begin{cases} \mathbf{0}, & \text{if } \mathbf{w}^\top \mathbf{h} \leq \alpha, \\ \frac{\alpha - \mathbf{w}^\top \mathbf{h}}{\|\mathbf{w}\|_2^2} \mathbf{w}, & \text{otherwise.} \end{cases}" />
          <p>
            We also evaluate a <strong>MERA logistic</strong> variant using logit probes. Steering is applied across mid-to-late layers following probe performance trends.
          </p>
        </section>

        {/* Results */}
        <section id="results">
          <h2>Results</h2>

          <h3 id="probe-results">Probe Quality Across Layers</h3>
          <p>
            We first assess how well probes predict model error as a function of transformer layer. Linear probes consistently achieve low RMSE in mid-to-late layers across all datasets and models, while logit probes exhibit higher variance and increased RMSE. The advantage of linear probes is most evident in intermediate layers, where accurate continuous error estimation is required for calibrated steering.
          </p>
        </section>

        <FigureCard
          number={1}
          title="Layer-wise RMSE: Linear vs. Logit Probes"
          caption="Comparison of linear and logit-based probes for error prediction across transformer layers, evaluated at the exact and last token positions on all 6 datasets. Each panel column is a dataset; each row is a token position. All 4 model variants are shown as lines; the dashed line is the dummy RMSE baseline. Click any panel to expand."
          interpretations={[
            "Linear probes consistently achieve lower RMSE than logit probes across datasets and layers.",
            "Error signals are strongest in mid-to-late layers (layers 12–28), consistent across model families.",
            "Llama-family models achieve lower RMSE than Apertus across most datasets.",
            "The dummy baseline (dashed) is flat, confirming probes learn non-trivial error structure.",
          ]}
          dataPath="/data/robustness.json"
          wide
        >
          <JsonChartLoader path="/data/robustness.json">
            {(data) => <LayerRMSEChart data={data as any} />}
          </JsonChartLoader>
        </FigureCard>

        <div className="article-prose">
          <h3>Cross-Dataset Generalisation</h3>
          <p>
            Cross-dataset evaluation reveals that error probes exhibit limited but structured generalisation. Probes trained and evaluated on the same dataset consistently outperform cross-dataset probes, confirming that error representations are partially domain-specific. However, datasets with similar task structures (e.g., MMLU High School and MMLU Professional) show relatively strong cross-generalisation. Probes trained on ARC-Easy generalise well to both ARC-Challenge and MMLU benchmarks, suggesting its mixture of factual recall and lightweight reasoning produces a more transferable error signal.
          </p>
        </div>

        <FigureCard
          number={2}
          title="Cross-Dataset Transfer Heatmap"
          caption="Average minimum RMSE (best layer performance, averaged across all models) when probes are trained on one dataset (rows) and evaluated on another (columns). Diagonal entries correspond to same-dataset evaluation. Darker colours indicate lower (better) predicted error."
          interpretations={[
            "Same-dataset probes consistently outperform cross-dataset probes (lowest RMSE on diagonal).",
            "MMLU HS ↔ MMLU Prof shows strong cross-generalisation, reflecting similar task structure.",
            "SMS Spam probes generalise poorly to reasoning-heavy datasets and vice versa.",
            "ARC-Easy probes transfer well to ARC-Challenge and MMLU benchmarks.",
          ]}
          dataPath="/data/transfer_rmse.json"
          wide
        >
          <JsonChartLoader path="/data/transfer_rmse.json">
            {(data) => <TransferHeatmap data={data as any} />}
          </JsonChartLoader>
        </FigureCard>

        <FigureCard
          number={3}
          title="Cross-Dataset Layer-wise RMSE Curves"
          caption="Layer-wise RMSE of linear error-estimation probes (L-0.01) evaluated at the exact token position. Each panel shows a different test dataset; curves correspond to different training datasets. Bold solid lines = same-dataset (diagonal); dashed = cross-dataset transfer. Use the model selector to switch between models."
          interpretations={[
            "Same-dataset probes (bold lines) consistently achieve the lowest RMSE.",
            "MMLU HS ↔ MMLU Prof shows strong cross-generalisation due to similar task structure.",
            "SMS Spam probes generalise poorly to reasoning-heavy datasets and vice versa.",
            "ARC-Easy probes transfer well to ARC-Challenge and MMLU benchmarks.",
            "Error signals are strongest in mid-to-late layers (12–24) across all pairs.",
          ]}
          dataPath="/data/transfer_rmse.json"
          wide
        >
          <JsonChartLoader path="/data/transfer_rmse.json">
            {(data) => <CrossDatasetLineChart data={data as any} />}
          </JsonChartLoader>
        </FigureCard>

        <div className="article-prose">
          <h3 id="steering-results">Steering Performance</h3>
          <p>
            We report steering results using the exact token position across all models and datasets. Performance is measured using Steering Performance Impact (SPI), a normalised metric bounded in <InlineMath math="[-1, 1]" /> that captures relative improvement scaled by the remaining performance headroom:
          </p>
          <BlockMath math="\text{SPI} = \begin{cases} \frac{\tilde{A}_{\mathcal{D}_{\text{test}}} - A_{\mathcal{D}_{\text{test}}}}{1 - A_{\mathcal{D}_{\text{test}}}}, & \text{if } \tilde{A}_{\mathcal{D}_{\text{test}}} > A_{\mathcal{D}_{\text{test}}}, \\ \frac{\tilde{A}_{\mathcal{D}_{\text{test}}} - A_{\mathcal{D}_{\text{test}}}}{A_{\mathcal{D}_{\text{test}}}}, & \text{otherwise.} \end{cases}" />
          <p>
            A value of SPI = 0 indicates no effect; positive values indicate improvement.
          </p>
          <p>
            A consistent pattern emerges: linear MERA achieves positive or neutral SPI on most datasets, while logit-based steering exhibits little to no improvement. Linear MERA delivers substantial gains on SMS Spam, where SPI values exceed 0.5 for several model variants. In contrast, performance on reasoning-heavy benchmarks (MMLU, ARC) remains closer to zero, suggesting limited improvement for steering in these settings.
          </p>
        </div>

        <FigureCard
          number={4}
          title="Steering Performance Impact by Method"
          caption="SPI grouped by model and dataset, comparing MERA, MERA logistic, MERA contrastive, prompt steering, and additive baselines. Top row: linear probes; bottom row: logit probes."
          interpretations={[
            "MERA consistently achieves positive or neutral SPI, never causing severe degradation.",
            "Contrastive steering yields strongly negative SPI on several datasets (below −0.8), highlighting its poor generalization.",
            "SMS Spam benefits most from steering across all models and probe variants.",
            "Instruction-tuned models show smaller SPI gains, consistent with higher baseline accuracy.",
          ]}
          dataPath="/data/spi_methods.json"
          wide
        >
          <JsonChartLoader path="/data/spi_methods.json">
            {(data) => <SPIBarChart data={data as any} />}
          </JsonChartLoader>
        </FigureCard>

        <FigureCard
          number={5}
          title="SPI Heatmap: Model × Dataset"
          caption="Per-model, per-dataset SPI at the exact token position. Each panel corresponds to a different steering method and probe type. Red indicates degradation; blue indicates improvement."
          interpretations={[
            "Linear MERA displays coherent positive structure, concentrated on SMS Spam.",
            "Contrastive steering exhibits sharp sign changes across datasets, indicating poor generalisation.",
            "Logit-based MERA shows a flat heatmap with low-magnitude values.",
            "Cross-domain contrastive vectors yield near-maximal negative SPI, underscoring its poor generalization.",
          ]}
          dataPath="/data/spi_heatmap.json"
          wide
        >
          <JsonChartLoader path="/data/spi_heatmap.json">
            {(data) => <SPIHeatmap data={data as any} />}
          </JsonChartLoader>
        </FigureCard>

        {/* Robustness */}
        <div className="article-prose">
          <h3>Mixed-Dataset Robustness</h3>
          <p>
            We investigate two extensions to assess robustness: mixed-dataset probe training and cross-domain steering. Training probes on a mixture of all datasets consistently improves error prediction quality across layers for both model families, as previously seen in Figure 1. Mixed-training probes achieve RMSE values substantially below the dummy baseline, indicating that the probe captures non-trivial error structure rather than dataset priors alone.
          </p>
          <p>
            For Apertus-Instruct, mixed training reduces RMSE relative to single-dataset probes across most layers, with the strongest gains in middle layers. The gap between Llama and Apertus narrows under mixed training, indicating that dataset diversity partially compensates for weaker baseline calibration. Increasing the probe learning rate from 0.02 to 0.05 leads to uniformly higher RMSE, suggesting that overfitting to heterogeneous error patterns degrades generalisation.
          </p>
        </div>

        <FigureCard
          number={6}
          title="Mixed-Dataset Probe Performance"
          caption="Layer-wise RMSE of probes trained on a mixture of all datasets and evaluated on the same mixture. Solid lines represent linear probes (L-0.01), while dashed lines represent logit probes (Logit-L-0.01)."
          interpretations={[
            "Mixed training yields robust error prediction across diverse distributions.",
            "Linear probes (solid) continue to decisively outperform logit probes (dashed) even in a mixed-dataset setting.",
            "RMSE is lowest in middle layers for mixed datasets, similar to single datasets."
          ]}
          dataPath="/data/probe_all_datasets_exact.json"
          wide
        >
          <JsonChartLoader path="/data/probe_all_datasets_exact.json">
            {(data) => <MixedProbeLineChart data={data as any} />}
          </JsonChartLoader>
        </FigureCard>

        <FigureCard
          number={7}
          title="Mixed-Dataset Steering SPI"
          caption="Steering Performance Impact (SPI) for mixed-dataset steering, where probes trained on a mixture of datasets are applied to individual target benchmarks. Grouped by target dataset, comparing MERA and baseline methods."
          interpretations={[
            "Mixed steering achieves robust positive SPI on SMS Spam even when it is only part of the training mixture.",
            "On reasoning benchmarks (MMLU, ARC), mixed steering is largely neutral.",
            "Mixed-trained probes avoid the severe degradations seen with contrastive and fixed-strength baselines.",
            "Mixed training trades peak per-task performance for cross-domain robustness.",
          ]}
          dataPath="/data/spi_methods.json"
          wide
        >
          <JsonChartLoader path="/data/spi_methods.json">
            {(data) => <SPIMixChart data={data as any} />}
          </JsonChartLoader>
        </FigureCard>

        {/* Limitations */}
        <section id="limitations" className="article-prose">
          <h2>Limitations</h2>
          <ul>
            <li>All experiments are conducted on 8B-parameter models. It remains unclear whether these findings transfer to larger or smaller architectures.</li>
            <li>We focus on multiple-choice and binary classification tasks. Open-ended generation tasks may exhibit different error structure.</li>
            <li>Steering relies on access to internal activations, which limits deployment in black-box settings.</li>
            <li>The SPI metric, while normalised, may mask absolute accuracy differences when baseline performance is very high or very low.</li>
            <li>Mixed-dataset steering trades peak per-task performance for robustness, which may not be desirable in all deployment scenarios.</li>
          </ul>
        </section>

        {/* Conclusion */}
        <section id="conclusion" className="article-prose">
          <h2>Conclusion &amp; Future Work</h2>
          <p>
            We reproduced and extended the MERA framework on the Apertus family of language models. Linear probes capture error-related signals consistently across layers, while logit-based probes are less stable and less informative. Calibrated linear steering improves performance on tasks where the base model performs poorly—especially on SMS Spam—and avoids the large performance drops seen with contrastive and fixed-strength baselines.
          </p>
          <p>
            Training probes on mixed datasets leads to more stable behaviour across domains, though the gains are smaller than task-specific steering. Mixed-domain steering shows that cross-domain probes can still produce positive effects, but improvements are weaker and more task-dependent. Overall, MERA is a reliable and safe method for inference-time error reduction, particularly when base model accuracy is low.
          </p>
          <p>
            <strong>Future work</strong> includes extending these experiments to larger models (70B+), evaluating on open-ended generation tasks, investigating non-linear probe architectures, and studying the interaction between activation steering and other inference-time techniques such as guided decoding.
          </p>
        </section>

        {/* Code Availability */}
        <section id="code" className="article-prose">
          <h2>Code Availability</h2>
          <p>
            Associated code is available open-source and under the MIT License, which
            permits free reuse, free modification, and free distribution, provided proper attribution
            is given to the authors, and no liability is assumed by the authors.
            The code and its MIT license are publicly available at{" "}
            <a
              href="https://github.com/swiss-ai/apertus-probes/"
              className="underline decoration-primary/40 underline-offset-2 hover:decoration-primary transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              github.com/swiss-ai/apertus-probes
            </a>.
          </p>
        </section>
      </article>

      {/* Footer */}
      <footer className="mt-16 pt-8 border-t font-sans text-xs text-muted-foreground">
        <p>
          Stepancic, Nguyen, Durech & Hedström. ETH Zurich, 2026. Preprint—not peer-reviewed.
        </p>
      </footer>
    </PaperLayout>
  );
}
