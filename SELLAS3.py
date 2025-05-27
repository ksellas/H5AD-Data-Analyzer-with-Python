import streamlit as st
import hdf5plugin
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import scanpy.external as sce
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>H5AD Data Analyzer</h1>", unsafe_allow_html=True)


tabs = st.tabs(["Ανάλυση", "Ομάδα"])


with tabs[0]:
    st.header("📂 Φόρτωση Δεδομένων")
    uploaded_file = st.file_uploader("Ανέβασε αρχείο .h5ad", type="h5ad")

    if uploaded_file:
        adata = sc.read_h5ad(uploaded_file)
        st.success(f"Το dataset περιέχει {adata.n_obs} κύτταρα και {adata.n_vars} γονίδια.")


        with st.expander("Φιλτράρισμα"):
            min_genes = st.slider("Ελάχιστα γονίδια ανά κύτταρο", 0, 1000, 200)
            max_genes = st.slider("Μέγιστα γονίδια ανά κύτταρο", 1000, 10000, 2500)
            min_cells = st.slider("Ελάχιστα κύτταρα ανά γονίδιο", 0, 100, 3)
            mito_prefix = st.text_input("Πρόθεμα για μιτοχονδριακά γονίδια (π.χ. MT-)", "MT-")

            if st.button("Εκτέλεση Φιλτραρίσματος"):
                adata.var['mt'] = adata.var_names.str.startswith(mito_prefix)
                sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
                sc.pp.filter_cells(adata, min_genes=min_genes)
                sc.pp.filter_genes(adata, min_cells=min_cells)
                adata = adata[adata.obs.n_genes_by_counts < max_genes, :]
                st.success("Ολοκληρώθηκε το φιλτράρισμα")

        with st.expander("Κανονικοποίηση και PCA"):
            if st.button("Κανονικοποίηση και PCA"):
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
                adata = adata[:, adata.var.highly_variable]
                sc.pp.scale(adata, max_value=10)
                sc.pp.calculate_qc_metrics(adata, inplace=True)
                sc.tl.pca(adata, svd_solver='arpack')
                st.success("Ολοκληρώθηκε η κανονικοποίηση και PCA")

                
                fig, ax = plt.subplots(figsize=(2, 1))
                sc.pl.pca(adata, color='n_genes_by_counts', show=False, ax=ax)
                ax.set_title(ax.get_title(), fontsize=5)
                ax.set_xlabel(ax.get_xlabel(), fontsize=3)
                ax.set_ylabel(ax.get_ylabel(), fontsize=3)
                ax.tick_params(axis='both', labelsize=2) 
                st.pyplot(fig)
                

        with st.expander("Clustering και UMAP"):
            obs_keys = list(adata.obs.columns)
            color_key = st.selectbox("Επίλεξε χαρακτηριστικό για χρωματισμό", options=obs_keys, index=0)
            use_harmony = st.checkbox("Χρήση Harmony για διόρθωση batch effects")
            if st.button("Clustering και UMAP"):
                
                adata = adata[:, [gene for gene in adata.var_names if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
                adata.raw = adata
                adata = adata[:, adata.var.highly_variable]
                sc.pp.scale(adata, max_value=10)
                sc.pp.pca(adata)

                if use_harmony:
                    import scanpy.external as sce
                    sce.pp.harmony_integrate(adata, 'batch')
                    sc.pp.neighbors(adata, use_rep="X_pca_harmony")
                else:
                    sc.pp.neighbors(adata)
                sc.tl.umap(adata)

                st.success("Ολοκληρώθηκε το clustering και UMAP")

        if "X_umap" in adata.obsm:
            st.subheader("UMAP Visualization")

            fig, ax = plt.subplots(figsize=(4, 3))
            sc.pl.umap(adata, color=color_key, show=False, ax=ax)
            ax.set_title(ax.get_title(), fontsize=5)
            ax.set_xlabel(ax.get_xlabel(), fontsize=5)
            ax.set_ylabel(ax.get_ylabel(), fontsize=5)
            ax.tick_params(axis='both', labelsize=5) 
            st.pyplot(fig)

        with st.expander("Differential Expression Analysis"):
            sc.tl.rank_genes_groups(
                adata,
                groupby='disease',          
                method='wilcoxon',           
                groups=['case'],           
                reference='control',        
                use_raw=False                
            )
            deg_result = adata.uns["rank_genes_groups"]

            degs_df = pd.DataFrame(
                {
                    "genes": deg_result["names"]["case"],
                    "pvals": deg_result["pvals"]["case"],
                    "pvals_adj": deg_result["pvals_adj"]["case"],
                    "logfoldchanges": deg_result["logfoldchanges"]["case"],
                }
            )
            degs_df

            degs_df["neg_log10_pval"] = -np.log10(degs_df["pvals"])

            
            degs_df["diffexpressed"] = "NS"
            degs_df.loc[(degs_df["logfoldchanges"] > 1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "UP"
            degs_df.loc[(degs_df["logfoldchanges"] < -1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "DOWN"

            
            top_downregulated = degs_df[degs_df["diffexpressed"] == "DOWN"]
            top_downregulated = top_downregulated.sort_values(by=["neg_log10_pval", "logfoldchanges"], ascending=[False, True]).head(20)

            
            top_upregulated = degs_df[degs_df["diffexpressed"] == "UP"]
            top_upregulated = top_upregulated.sort_values(by=["neg_log10_pval", "logfoldchanges"], ascending=[False, False]).head(81)

           
            top_genes_combined = pd.concat([top_downregulated["genes"], top_upregulated["genes"]])
            df_annotated = degs_df[degs_df["genes"].isin(top_genes_combined)]

            
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=degs_df, x="logfoldchanges", y="neg_log10_pval", hue="diffexpressed", palette={"UP": "#bb0c00", "DOWN": "#00AFBB", "NS": "grey"}, alpha=0.7, edgecolor=None)

            
            plt.axhline(y=-np.log10(0.05), color='gray', linestyle='dashed')
            plt.axvline(x=-1, color='gray', linestyle='dashed')
            plt.axvline(x=1, color='gray', linestyle='dashed')

        
            plt.xlim(-11, 11)
            plt.ylim(25, 175)
            plt.xlabel("log2 Fold Change", fontsize=14)
            plt.ylabel("-log10 p-value", fontsize=14)
            plt.title("Volcano of DEGs (Disease vs Control)", fontsize=16)
            plt.legend(title="Expression", loc="upper right")

            st.pyplot()

        if st.button("Αποθήκευση αποτελεσμάτων"):
            adata.write("processed_data.h5ad")
            st.success("Αποθηκεύτηκε ως processed_data.h5ad")

with tabs[1]:
    st.header("Πληροφορίες Ομάδας")
    st.markdown("""
    **Μέλη Ομάδας:**
    - Σελλάς Κωνσταντίνος (ΑΜ: inf2022184)
    - Αλεξάνδρου Αγγελος (ΑΜ: inf2022004)

    **Συνεισφορά:**
    - Ανάπτυξη Streamlit εφαρμογής
    - Ολοκλήρωση βημάτων preprocessing, clustering
    - Οπτικοποίηση αποτελεσμάτων με UMAP
    - GitHub upload
    """)